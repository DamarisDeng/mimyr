from anndata import AnnData
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
from .utils.hf_tokenizer import MimyrTokenizer
from .Mimyr import Mimyr, fine_tuning, generate_prompt_for_cg
import inspect
from typing import Tuple, List, Optional
import pickle as pkl
import scipy.sparse as sp

# from Mimyr import generate_cellGenesis

import pandas as pd
import numpy as np
import scanpy as sc
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class _AnndataGenDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        meta_pool: dict,
        sep_token: str,
        tokenizer: MimyrTokenizer,
        max_len: int,
        scm: Mimyr,
        include_0s: bool,
        add_xyz_noise: bool,
        min_max_bounds: List[int],
        exclude_columns: List[str] = None,
    ):
        """
        Expects:
          - adata.obs has whatever columns generate_prompt_for_cg() needs
          - adata.obs['gene_tokens'] is a list of gene‐token strings per cell
        """
        self.adata = adata
        self.meta_pool = meta_pool
        self.sep_token = sep_token
        self.tok = tokenizer
        self.max_len = max_len
        self.scm = scm
        self.include_0s = include_0s
        self.add_xyz_noise = add_xyz_noise
        self.min_max_bounds = min_max_bounds
        self.exclude_columns = exclude_columns

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 1) build the prompt
        prompt_tokens, prompt_vals = generate_prompt_for_cg(
            idx,
            self.adata.obs,
            self.meta_pool,
            self.tok,
            self.add_xyz_noise,
            self.min_max_bounds,
            self.exclude_columns,
        )
        prompt_tokens += self.tok.encode([self.sep_token])
        prompt_vals += [0]

        target_tokens, target_vals, target_real_vals = (
            self.scm.get_gene_and_expression_tokens(idx, self.include_0s)
        )
        # 3) full token sequence
        ids = prompt_tokens + target_tokens + self.tok.encode(["<E>"])
        vals = prompt_vals + list(target_vals) + [0]
        target_real_vals = prompt_vals + list(target_real_vals) + [0]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + self.tok.encode(["<E>"])
            vals = vals[: self.max_len - 1] + [0]
            target_real_vals = target_real_vals[: self.max_len - 1] + [0]
        pad_len = self.max_len - len(ids)
        ids += [self.tok.pad_token_id] * pad_len
        vals += [0] * pad_len
        target_real_vals += [0] * pad_len

        # 5) labels: ignore prompt
        labels = [-100] * len(prompt_tokens) + target_tokens + self.tok.encode(["<E>"])
        if len(labels) > self.max_len:
            labels = labels[: self.max_len - 1] + self.tok.encode(["<E>"])
        labels += [-100] * (self.max_len - len(labels))
        # ids += [-100] * (self.max_len - len(ids))
        # vals += [-100] * (self.max_len - len(vals))

        # 6) attention mask
        mask = [1 if i != self.tok.pad_token_id else 0 for i in ids]
        assert len(labels) <= self.max_len
        assert len(ids) <= self.max_len
        assert len(vals) <= self.max_len

        input_ids = torch.tensor(ids, dtype=torch.long).clone()
        labels = torch.tensor(labels, dtype=torch.long).clone()
        attention_mask = torch.tensor(mask, dtype=torch.long).clone()
        input_vals = torch.tensor(vals, dtype=torch.long).clone()
        target_vals = torch.tensor(target_real_vals, dtype=torch.float32).clone()
        # print(input_ids.shape, labels.shape, attention_mask.shape, input_vals.shape, target_vals.shape)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_vals": input_vals,
            "target_vals": target_vals,
            "idx": idx,
        }


# --- helper 1: reproducible multinomial draw over cells ---
def _draw_indices_by_prob(
    n_cells: int, probs: np.ndarray, num_samples: int, seed: int = 0
) -> np.ndarray:
    """Draw `num_samples` indices with replacement using `probs` (sum=1)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_cells, size=int(num_samples), replace=True, p=probs)
    return idx.astype(np.int64)


# --- helper 2: a simple sampler over a fixed list of indices ---
class SubsetSampler(Sampler):
    def __init__(self, indices: np.ndarray):
        self.indices = indices

    def __iter__(self):
        # Already pre-shuffled by us; yield in given order
        return iter(self.indices.tolist())

    def __len__(self):
        return self.indices.shape[0]


class EpochWeightedSampler(torch.utils.data.Sampler):
    """
    Single-GPU: draws 'epoch_samples' indices with replacement each epoch from probs.
    Call .set_epoch(epoch) in your training loop.
    """

    def __init__(
        self, n_cells: int, probs: np.ndarray, epoch_samples: int, seed: int = 0
    ):
        self.n = int(n_cells)
        self.m = int(epoch_samples)
        self.seed = int(seed)
        p = np.asarray(probs, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        self.p = p / p.sum()
        self.epoch = 0
        self.last_indices_ = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng((self.seed + self.epoch) & 0xFFFFFFFF)
        idx = rng.choice(self.n, size=self.m, replace=True, p=self.p)
        rng.shuffle(idx)
        self.last_indices_ = idx
        return iter(idx.tolist())

    def __len__(self):
        return self.m

    def get_last_indices(self):
        return np.array([]) if self.last_indices_ is None else self.last_indices_.copy()


class EpochWeightedSamplerDDP(torch.utils.data.Sampler):
    """
    DDP-aware: draws a global index list, then shards by rank via striding.
    """

    def __init__(
        self, n_cells: int, probs: np.ndarray, epoch_samples: int, seed: int = 0
    ):
        assert dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.base = EpochWeightedSampler(n_cells, probs, epoch_samples, seed)
        self.last_indices_ = None

    def set_epoch(self, epoch: int):
        self.base.set_epoch(epoch)

    def __iter__(self):
        # draw global, then shard by stride
        global_iter = iter(self.base)
        global_idx = np.fromiter(global_iter, dtype=np.int64)  # materialize once
        local_idx = global_idx[self.rank :: self.world_size]
        self.last_indices_ = local_idx
        return iter(local_idx.tolist())

    def __len__(self):
        # approximate per-rank length
        return int(np.ceil(self.base.m / self.world_size))

    def get_last_indices(self):
        return np.array([]) if self.last_indices_ is None else self.last_indices_.copy()


def _full_dist(adata, key: str):
    s = adata.obs[key].astype("category")
    cnt = s.value_counts()
    frac = cnt / cnt.sum()
    return cnt, frac


def _sample_dist(adata, idx: np.ndarray, key: str):
    s = adata.obs[key].astype("category").iloc[idx]
    cnt = s.value_counts()
    frac = cnt / cnt.sum()
    return cnt, frac


def summarize_sample_singlegpu(
    adata, idx: np.ndarray, cell_key="cell_type", group_key="dataset_type", top_k=20
) -> None:
    """Print top_k classes by sampled count, with lift vs. full, and dataset_type mix."""
    # Cell-type table
    full_c, full_f = _full_dist(adata, cell_key)
    samp_c, samp_f = _sample_dist(adata, idx, cell_key)
    df = pd.DataFrame(
        {
            "full_count": full_c,
            "full_frac": full_f,
            "samp_count": samp_c,
            "samp_frac": samp_f,
        }
    ).fillna(0.0)
    df["lift"] = (df["samp_frac"] / df["full_frac"]).replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("samp_count", ascending=False).head(top_k)
    print("\n=== Sampled cell types (top {}) ===".format(top_k))
    print(df.to_string(float_format=lambda x: f"{x:,.4f}"))

    # Dataset_type table (if present)
    if group_key in adata.obs.columns:
        fc, ff = _full_dist(adata, group_key)
        scnt, sfrac = _sample_dist(adata, idx, group_key)
        gdf = pd.DataFrame(
            {"full_count": fc, "full_frac": ff, "samp_count": scnt, "samp_frac": sfrac}
        ).fillna(0.0)
        gdf["lift"] = (gdf["samp_frac"] / gdf["full_frac"]).replace(
            [np.inf, -np.inf], np.nan
        )
        print("\n=== Sampled dataset_type ===")
        print(gdf.to_string(float_format=lambda x: f"{x:,.4f}"))


def _merge_counts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def summarize_sample_ddp(
    adata,
    local_idx: np.ndarray,
    cell_key="cell_type",
    group_key="dataset_type",
    top_k=20,
) -> None:
    """
    Aggregate per-rank counts to rank 0 and print global stats.
    """
    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1

    # Build local counts as dicts (to gather easily)
    ct_local = (
        adata.obs[cell_key].astype("category").iloc[local_idx].value_counts().to_dict()
    )
    if group_key in adata.obs.columns:
        grp_local = (
            adata.obs[group_key]
            .astype("category")
            .iloc[local_idx]
            .value_counts()
            .to_dict()
        )
    else:
        grp_local = {}

    objs = [ct_local, grp_local]
    if ddp:
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, objs)
    else:
        gathered = [objs]

    # Merge across ranks
    ct_all = _merge_counts([g[0] for g in gathered])
    grp_all = _merge_counts([g[1] for g in gathered])

    if rank == 0:
        # Cell-type table
        full_c, full_f = _full_dist(adata, cell_key)
        samp_c = pd.Series(ct_all).astype(int)
        samp_c = samp_c[samp_c.index.isin(full_c.index)]
        samp_f = samp_c / samp_c.sum()
        df = pd.DataFrame(
            {
                "full_count": full_c,
                "full_frac": full_f,
                "samp_count": samp_c,
                "samp_frac": samp_f,
            }
        ).fillna(0.0)
        df["lift"] = (df["samp_frac"] / df["full_frac"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df = df.sort_values("samp_count", ascending=False).head(top_k)
        print("\n=== [DDP] Sampled cell types (top {}) ===".format(top_k))
        print(df.to_string(float_format=lambda x: f"{x:,.4f}"))

        # Dataset_type table
        if group_key in adata.obs.columns and len(grp_all) > 0:
            fc, ff = _full_dist(adata, group_key)
            scnt = pd.Series(grp_all).astype(int)
            scnt = scnt[scnt.index.isin(fc.index)]
            sfrac = scnt / scnt.sum()
            gdf = pd.DataFrame(
                {
                    "full_count": fc,
                    "full_frac": ff,
                    "samp_count": scnt,
                    "samp_frac": sfrac,
                }
            ).fillna(0.0)
            gdf["lift"] = (gdf["samp_frac"] / gdf["full_frac"]).replace(
                [np.inf, -np.inf], np.nan
            )
            print("\n=== [DDP] Sampled dataset_type ===")
            print(gdf.to_string(float_format=lambda x: f"{x:,.4f}"))


def get_generation_dataloader(
    adata: AnnData,
    meta_info: dict,
    batch_size: int = 8,
    max_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    include_0s: bool = True,
    n_express_level: int = 10,
    add_xyz_noise: bool = False,
    exclude_columns: Optional[List[str]] = None,
    # --- new knobs for weighted sampling ---
    use_sampling_probs: bool = True,  # respect adata.obs['sampling_prob'] if present
    sampling_col: str = "sampling_prob",
    epoch_samples: Optional[
        int
    ] = 1000000,  # how many rows per epoch; default = len(ds)
    seed: int = 0,  # base seed for draws
    bin_edges=None,
) -> DataLoader:
    """
    Build a DataLoader for conditional-generation that can sample cells
    according to precomputed per-cell probabilities in `adata.obs[sampling_col]`.
    Works in single-GPU and DDP. If DDP is initialized, indices are sharded by rank.
    """
    # 1) tokenizer
    tokenizer = MimyrTokenizer(meta_info["token_set"])
    tokenizer.add_special_tokens(
        {"sep_token": meta_info.get("sep_token", "<SPToken1>")}
    )
    tokenizer.pad_token = "<SPToken10>"

    # 2) SCM preproc (unchanged)
    scm = fine_tuning(
        adata, meta_info, n_express_level=n_express_level, bin_edges=bin_edges
    )
    scm.data_preprocess()

    bin_edges = scm.bin_edges

    # 3) dataset (unchanged)
    ds = _AnndataGenDataset(
        adata,
        meta_pool=meta_info,
        sep_token=meta_info.get("sep_token", "<SPToken1>"),
        tokenizer=tokenizer,
        max_len=max_len,
        scm=scm,
        include_0s=include_0s,
        add_xyz_noise=add_xyz_noise,
        min_max_bounds=[0, n_express_level - 1],
        exclude_columns=exclude_columns,
    )

    N = len(ds)
    # If caller didn't specify, default to drawing as many rows as dataset length
    if epoch_samples is None:
        epoch_samples = N

    # 4) Decide on sampler
    sampler = None
    dl_shuffle = False  # we'll control shuffling ourselves when we pre-draw

    can_use_probs = (
        use_sampling_probs
        and sampling_col in adata.obs.columns
        and np.isfinite(adata.obs[sampling_col].values).all()
        and adata.obs[sampling_col].values.sum() > 0
    )

    if can_use_probs:

        probs = adata.obs[sampling_col].to_numpy(dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs /= probs.sum()

        if dist.is_available() and dist.is_initialized():
            sampler = EpochWeightedSamplerDDP(
                n_cells=N, probs=probs, epoch_samples=epoch_samples, seed=seed
            )
        else:
            sampler = EpochWeightedSampler(
                n_cells=N, probs=probs, epoch_samples=epoch_samples, seed=seed
            )

        dl_shuffle = False

    else:
        # --- Fallback: classic (Distributed)Sampler with optional shuffle ---
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle, drop_last=False, seed=seed
            )
            dl_shuffle = False
        else:
            sampler = None
            dl_shuffle = shuffle  # vanilla shuffle only if no sampler

    return (
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=dl_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            generator=torch.Generator().manual_seed(
                seed
            ),  # deterministic collation order
        ),
        bin_edges,
    )


# def get_generation_dataloader(
#     adata: AnnData,
#     meta_info: dict,
#     batch_size: int = 8,
#     max_len: int = 512,
#     shuffle: bool = True,
#     num_workers: int = 4,
#     include_0s: bool = True,
#     n_express_level: int = 10,
#     add_xyz_noise: bool = False,
#     exclude_columns: List[str] = None,
# ) -> DataLoader:
#     """
#     Build a DataLoader for fine‑tuning on the conditional‑generation task.

#     Parameters
#     ----------
#     adata : AnnData
#         Your AnnData with `.obs` containing both the fields that
#         `generate_prompt_for_cg` needs and a column
#         `adata.obs['gene_tokens']`, where each entry is a list of
#         gene‐token strings for that cell.
#     meta_info : dict
#         Loaded from your `meta_info.pt`. Must contain:
#           - 'token_set'   : list of all tokens
#           - 'sep_token'   : e.g. '<SPToken1>'
#           - 'meta_pool'   : whatever generate_prompt_for_cg expects
#     batch_size, max_len, shuffle, num_workers : as usual

#     Returns
#     -------
#     DataLoader yielding dicts with keys 'input_ids', 'labels', 'attention_mask', 'expression'
#     """
#     # 1) tokenizer
#     tokenizer = MimyrTokenizer(meta_info['token_set'])
#     tokenizer.add_special_tokens({'sep_token': meta_info.get('sep_token', '<SPToken1>')})
#     tokenizer.pad_token = '<SPToken10>'

#     scm = fine_tuning(adata, meta_info, n_express_level=n_express_level)
#     scm.data_preprocess()

#     # 2) dataset
#     ds = _AnndataGenDataset(
#         adata,
#         meta_pool = meta_info,
#         sep_token = meta_info.get('sep_token', '<SPToken1>'),
#         tokenizer = tokenizer,
#         max_len   = max_len,
#         scm   = scm,
#         include_0s = include_0s,
#         add_xyz_noise = add_xyz_noise,
#         min_max_bounds = [0,n_express_level - 1],
#         exclude_columns = exclude_columns,
#     )

#     # 3) dataloader
#     if dist.is_available() and dist.is_initialized():
#         sampler = DistributedSampler(ds)
#     else:
#         sampler = None

#     return DataLoader(
#         ds,
#         batch_size  = batch_size,
#         shuffle=(sampler is None and shuffle),  # only shuffle if no sampler
#         sampler     = sampler,
#         num_workers = num_workers,
#         pin_memory  = True,
#     )


def harmonize_dataset(
    adata,
    meta_info,
    coordfiles,
    organ="Brain",
    technology="M550",
    coord_suffix="_ccf",
    n_bins=100,
    overwrite_technology=False,
):
    if adata.n_obs > 0 and adata.n_vars > 0 and adata.X.max() > 10:
        # base_per_gene = 5  # tune (e.g., 5–50)
        # n_panel_genes = adata.shape[1]  # after your consistent filtering
        # panel_target = base_per_gene * n_panel_genes

        # sc.pp.normalize_total(adata, target_sum=panel_target)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.filter_cells(adata, min_genes=10)
    if "<x>" not in adata.obs.columns:
        coord_bins = {}
        for coord, coordfile in zip(("x", "y", "z"), coordfiles):
            vals_full = adata.obs[f"{coord}{coord_suffix}"].values.astype(float)
            vals = adata.obs[f"{coord}{coord_suffix}"].dropna().values.astype(float)
            coord_bins[f"{coord}{coord_suffix}"] = np.linspace(
                vals.min(), vals.max(), n_bins
            )
            edges = pkl.load(open(coordfile, "rb"))
            # edges   = coord_bins[f'{coord}{coord_suffix}']
            bin_idxs = np.digitize(vals_full, edges, right=True)
            adata.obs[f"<{coord}>"] = bin_idxs
    if "organ" not in adata.obs.columns:
        adata.obs["organ"] = organ
    if "technology" not in adata.obs.columns or overwrite_technology:
        adata.obs["technology"] = technology
    cols = [
        "<x>",
        "<y>",
        "<z>",
        "organ",
        "class",
        "subclass",
        "supertype",
        "cluster",
        "technology",
    ]
    mask = adata.obs[cols].notnull().all(axis=1)
    adata = adata[mask].copy()

    existing_genes = adata.var_names.tolist()

    gene_set = list(meta_info["gene_set"])
    new_genes = [g for g in gene_set if g not in existing_genes]

    print(f"Adding {len(new_genes)} new genes to adata")

    if len(new_genes) > 0:
        n_obs = adata.n_obs
        n_new_vars = len(new_genes)

        new_vars = pd.DataFrame(index=new_genes)
        new_var = pd.concat([adata.var, new_vars], axis=0)

        if sp.issparse(adata.X):
            new_data = sp.csr_matrix((n_obs, n_new_vars))
            X = sp.hstack([adata.X, new_data], format="csr")
        else:
            new_data = np.zeros((n_obs, n_new_vars))
            X = np.hstack([adata.X, new_data])

        new_adata = sc.AnnData(
            X=X, var=new_var, obs=adata.obs, obsm=adata.obsm, uns=adata.uns
        )
        new_adata.var_names_make_unique()
        del adata
        adata = new_adata
    mask_nonempty = (
        (adata[:, gene_set].X.sum(axis=1) > 0).A1
        if sp.issparse(adata.X)
        else (adata[:, gene_set].X.sum(axis=1) > 0)
    )
    adata = adata[mask_nonempty].copy()

    return adata
