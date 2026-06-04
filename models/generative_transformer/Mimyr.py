import torch
import os
import sys
from .model.model import MimyrConfig, MimyrModel
from .model.model_kvcache import MimyrModel_kv
import torch.nn.functional as F
from .utils.hf_tokenizer import MimyrTokenizer
import scipy.sparse
import numpy as np
from tqdm import tqdm
from anndata import AnnData
from typing import Optional, List, Tuple, Union
import pandas as pd
import io
import multiprocessing
from tqdm import tqdm
import time

multiprocessing.set_start_method("spawn", force=True)


class Mimyr:

    def __init__(
        self,
        adata: AnnData,
        meta_info: dict,
        tokenizer,
        n_express_level: int,
        bin_edges: Optional[np.ndarray] = None,
        model: Optional[MimyrModel] = None,
        **kwargs,
    ):
        if model is not None:
            self.model = model
        self.adata = adata
        self.meta_info = meta_info
        self.tokenizer = tokenizer
        self.n_express_level = n_express_level
        if bin_edges is not None:
            self.bin_edges = torch.tensor(bin_edges)
        self.Mimyr_gene_set = meta_info["gene_set"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.check_adata(adata, **kwargs)

    def data_preprocess(
        self,
    ):

        self.adata_matrix = self.adata.X

    def get_gene_expression_dict(self, i, matrix):
        if scipy.sparse.issparse(matrix):
            row = matrix.getrow(i).toarray().ravel()
        else:
            row = matrix[i]
        expressed_idx = np.where(row > 0)[0]
        expressed_genes = np.array(self.Mimyr_gene_set)[expressed_idx]
        expr_values = row[expressed_idx]
        return dict(zip(expressed_genes, expr_values))

    def get_gene_expression_dict_with_0s(self, i, matrix):
        if scipy.sparse.issparse(matrix):
            row = matrix.getrow(i).toarray().ravel()
        else:
            row = matrix[i]
        expressed_genes = self.Mimyr_gene_set
        expr_values = row
        return dict(zip(expressed_genes, expr_values))

    def prepare_gene_expression_codings(self, i, matrix, include_0s=False):
        # 1) build the dict of gene→expr, possibly including zeros
        if include_0s:
            cell_expression_dict = self.get_gene_expression_dict_with_0s(i, matrix)
        else:
            cell_expression_dict = self.get_gene_expression_dict(i, matrix)

        # preserve your original ordering
        expressed_genes = list(cell_expression_dict.keys())[::-1]
        expression_values = np.array(
            list(cell_expression_dict.values())[::-1], dtype=float
        )

        # 2) identify zeros and positives
        zero_mask = expression_values == 0
        nonzero_vals = expression_values[~zero_mask]

        # 3) compute bins *only* on the positive values
        if len(nonzero_vals) > 0:
            # digitize positives into 1…N; clamp to [1, n_express_level-1] so
            # values above the training-data max don't exceed the embedding size
            pos_bins = np.digitize(nonzero_vals, self.bin_edges, right=True)
            pos_bins = np.clip(pos_bins, 1, self.n_express_level - 1)
        else:
            pos_bins = np.array([], dtype=int)

        # 4) merge back into a full-length array, with zeros→bin 0
        binned_expr = np.empty_like(expression_values, dtype=int)
        # fill zeros
        binned_expr[zero_mask] = 0
        # fill positives
        binned_expr[~zero_mask] = pos_bins
        # print(len(expressed_genes), binned_expr.shape, expression_values.shape)
        return expressed_genes, binned_expr, expression_values

    def make_encoded_gene_expression_one_cell(self, expressed_genes, binned_expr):

        prefix = expressed_genes
        ec_binned_expr = binned_expr
        ec_prefix = self.tokenizer.encode(prefix)

        return (ec_prefix, ec_binned_expr)

    def get_gene_and_expression_tokens(self, i, include_0s=True):

        expressed_genes, binned_expr, expression_vals = (
            self.prepare_gene_expression_codings(
                i, self.adata_matrix, include_0s=include_0s
            )
        )
        gene_tokens, expression_tokens = self.make_encoded_gene_expression_one_cell(
            expressed_genes, binned_expr
        )

        return (gene_tokens, expression_tokens, expression_vals)

    @torch.no_grad()
    def generate_cell_genesis(
        self,
        idx: Union[int, List[int]],
        max_new_tokens: int = 50,
        top_k: int = 5,
        return_gt: bool = False,
        cheat_with_tokens: bool = False,
        cheat_with_expr: bool = False,
        batch_size: int = 12,
        verbose: bool = False,
        fast=False,
        select_max_index_margin: float = 0.01,
        **generate_kwargs,
    ):
        """
        Generate gene-token sequences for idx via mini-batches of generate_cellGenesis calls.

        Parameters
        ----------
        idx : int or List[int]
            Single index or list of indices from adata.obs
        batch_size : int
            Number of cells to process per model call
        verbose : bool
            If True, prints timing for each major step

        Returns
        -------
        List of generated results per cell
        """
        self.data_preprocess()

        if isinstance(idx, int):
            idx = [idx]

        results = []
        for start in tqdm(range(0, len(idx), batch_size)):
            batch_start_time = time.time()
            batch_indices = idx[start : start + batch_size]

            if verbose:
                print(
                    f"\nProcessing batch {start // batch_size + 1} / {((len(idx) - 1) // batch_size) + 1}"
                )

            # === Prompt Construction ===
            t0 = time.time()
            prompt_ids_list = []
            prompt_vals_list = []
            target_tokens_list = []
            target_vals_list = []
            target_real_vals_list = []

            sep_id = self.tokenizer.convert_tokens_to_ids(
                self.meta_info.get("sep_token", "<SPToken1>")
            )

            for i in batch_indices:
                prompt_ids, prompt_vals = generate_prompt_for_cg(
                    i, self.adata.obs, self.meta_info, self.tokenizer
                )
                prompt_ids.append(sep_id)
                prompt_vals.append(0)

                prompt_ids_list.append(prompt_ids)
                prompt_vals_list.append(prompt_vals)

            if verbose:
                print(f" sep id is {sep_id}")
                print(f"prompt ids are: {prompt_ids_list}")
                print(f"  Prompt construction took {time.time() - t0:.2f} sec")

            # === Target Sequence Construction (optional) ===
            if return_gt or cheat_with_tokens or cheat_with_expr:
                t1 = time.time()
                for i, prompt_ids, prompt_vals in zip(
                    batch_indices, prompt_ids_list, prompt_vals_list
                ):
                    tgt_tokens, tgt_vals, tgt_real_vals = (
                        self.get_gene_and_expression_tokens(i, include_0s=False)
                    )
                    tgt_tokens = (
                        prompt_ids + tgt_tokens + self.tokenizer.encode(["<E>"])
                    )
                    tgt_vals = prompt_vals + list(tgt_vals) + [0]
                    tgt_real_vals = prompt_vals + list(tgt_real_vals) + [0]

                    target_tokens_list.append(tgt_tokens)
                    target_vals_list.append(tgt_vals)
                    target_real_vals_list.append(tgt_real_vals)
                if verbose:
                    print(f"target tokens are: {target_tokens_list}")
                    print(f"  Target sequence creation took {time.time() - t1:.2f} sec")

            # === Tensor Preparation ===
            t2 = time.time()
            max_len = max(len(p) for p in prompt_ids_list)
            pad_token = 0

            input_ids = torch.full(
                (len(batch_indices), max_len),
                pad_token,
                dtype=torch.long,
                device=self.device,
            )
            input_vals = torch.full(
                (len(batch_indices), max_len),
                pad_token,
                dtype=torch.long,
                device=self.device,
            )

            for b, (p_ids, p_vals) in enumerate(zip(prompt_ids_list, prompt_vals_list)):
                plen = len(p_ids)
                input_ids[b, :plen] = torch.tensor(
                    p_ids, dtype=torch.long, device=self.device
                )
                input_vals[b, :plen] = torch.tensor(
                    p_vals, dtype=torch.long, device=self.device
                )

            override_gene_sequence = None
            override_expr_sequence = None
            if return_gt or cheat_with_tokens or cheat_with_expr:
                max_target_len = max(len(t) for t in target_tokens_list)
                override_gene_sequence = torch.full(
                    (len(batch_indices), max_target_len),
                    pad_token,
                    dtype=torch.long,
                    device=self.device,
                )
                override_expr_sequence = torch.full(
                    (len(batch_indices), max_target_len),
                    pad_token,
                    dtype=torch.long,
                    device=self.device,
                )

                for b, (t_ids, t_vals) in enumerate(
                    zip(target_tokens_list, target_vals_list)
                ):
                    tlen = len(t_ids)
                    override_gene_sequence[b, :tlen] = torch.tensor(
                        t_ids, dtype=torch.long, device=self.device
                    )
                    override_expr_sequence[b, :tlen] = torch.tensor(
                        t_vals, dtype=torch.long, device=self.device
                    )

            if verbose:
                # print(f' input ids are {input_ids}')
                # print(f' override token sequence is {override_gene_sequence}')
                print(f"override expression sequence is {override_expr_sequence}")
                print(f"  Tensor preparation took {time.time() - t2:.2f} sec")

            # === Model Call ===
            t3 = time.time()
            if fast:
                generated_ids, generated_vals_binned, generated_vals = (
                    self.model.generate_cellGenesis_fast(
                        input_ids=input_ids,
                        expression_level=input_vals,
                        max_new_tokens=max_new_tokens + input_ids.shape[1],
                        top_k=top_k,
                        override_gene_sequence=(
                            override_gene_sequence if cheat_with_tokens else None
                        ),
                        override_expr_sequence=(
                            override_expr_sequence if cheat_with_expr else None
                        ),
                        verbose=verbose,
                        select_max_index_margin=select_max_index_margin,
                        **generate_kwargs,
                    )
                )
            else:
                generated_ids, generated_vals_binned, generated_vals = (
                    self.model.generate_cellGenesis(
                        input_ids=input_ids,
                        expression_level=input_vals,
                        max_new_tokens=max_new_tokens + input_ids.shape[1],
                        top_k=top_k,
                        override_gene_sequence=(
                            override_gene_sequence if cheat_with_tokens else None
                        ),
                        override_expr_sequence=(
                            override_expr_sequence if cheat_with_expr else None
                        ),
                        verbose=verbose,
                        select_max_index_margin=select_max_index_margin,
                        **generate_kwargs,
                    )
                )
            if verbose:
                print(f"  Model generation took {time.time() - t3:.2f} secondss")

            # === Post-Processing ===
            t4 = time.time()
            for b in range(len(batch_indices)):
                plen = len(prompt_ids_list[b])
                gen_seq = generated_ids[b].tolist()
                gen_vals = generated_vals[b].tolist()
                gen_vals_binned = generated_vals_binned[b].tolist()

                new_ids = gen_seq  # [plen:]
                new_vals = gen_vals  # [plen:]
                gene_tokens = self.tokenizer.convert_ids_to_tokens(new_ids)

                row = tokens_and_vals_to_expression_row(
                    var_names=self.adata.var_names.tolist(),
                    gene_tokens=gene_tokens,
                    gene_tokens_int=new_ids,
                    new_vals=new_vals,
                    return_series=False,
                )

                if return_gt:
                    tgt_tokens = torch.tensor(
                        target_tokens_list[b], dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    tgt_vals = torch.tensor(
                        target_vals_list[b], dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    tgt_real_vals = torch.tensor(
                        target_real_vals_list[b],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    results.append(
                        (
                            row,
                            gene_tokens,
                            new_vals,
                            gen_seq,
                            gen_vals_binned,
                            gen_vals,
                            tgt_tokens,
                            tgt_vals,
                            tgt_real_vals,
                        )
                    )
                else:
                    results.append(
                        (row, gene_tokens, new_vals, gen_seq, gen_vals_binned, gen_vals)
                    )
            if verbose:
                print(f"  Post-processing took {time.time() - t4:.2f} sec")
                print(f"Total batch time: {time.time() - batch_start_time:.2f} sec")

        return results

    def check_adata(
        self, adata, force=False
    ):  # set force as True to pass check adata anyway.

        if force:
            print("✅ forcing pass check")
            print(" Mimyr is ready")
            self.adata = adata.copy()
            return True
        # check normalize and log1p
        adata_max = adata.X.max()
        assert (
            adata_max < 10
        ), f"🚫 Please make sure adata is processed with normalization (sum = 1e4) and log1p, your adata max is {adata_max}."
        # check gene symbol uniform
        adata_var = set(adata.var_names.tolist())
        Mimyr_geneset = set(self.meta_info["gene_set"])
        count = len(adata_var.intersection(Mimyr_geneset))
        assert count == len(
            self.meta_info["gene_set"]
        ), f"🚫 Please make sure adata is processed with uniformed gene symbol, your gene set has {count} overlap with Mimyr."
        # use Mimyr gene set
        # self.adata = adata[:,self.Mimyr_gene_set].copy()
        self.adata = adata
        indices = [self.adata.var_names.get_loc(g) for g in self.Mimyr_gene_set]
        self.adata._inplace_subset_var(indices)
        print("✅ adata passed check")
        print("Mimyr is ready")


def model_inference(
    ckp_path: str,
    adata: AnnData,
    meta_info: dict,
    use_kv_cache: Optional[bool] = False,
    **kwargs,
):

    ckp = torch.load(ckp_path, map_location="cpu")
    gptconf = MimyrConfig(**ckp["model_args"])
    if use_kv_cache:
        model = MimyrModel_kv(gptconf)
    else:
        model = MimyrModel(gptconf)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(ckp["model"])
    model.eval()
    model.hidden_dim = ckp["model_args"]["n_embd"]
    tokenizer = MimyrTokenizer(meta_info["token_set"])
    n_express_level = ckp["model_args"]["expression_level"]

    scml = Mimyr(adata, meta_info, tokenizer, n_express_level, model=model, **kwargs)

    return scml


def model_generate(
    ckp_path: str,
    adata: AnnData,
    meta_info: dict,
    use_kv_cache: bool = False,
    n_express_level: int = 100,
    **kwargs,
):

    ckp = torch.load(ckp_path, map_location="cpu")
    # ckp['model_args']['expression_level'] = 100
    gptconf = MimyrConfig(**ckp["model_args"])
    gptconf.vocab_size = len(meta_info["token_set"])
    # bin_edges = compute_global_bin_edges(adata, meta_info['gene_set'],gptconf.expression_level)
    bin_edges = compute_global_bin_edges(adata, meta_info["gene_set"], n_express_level)
    gptconf.bin_edges = bin_edges
    gptconf.dropout = 0.0
    # print(gptconf)

    if use_kv_cache == False:
        model = MimyrModel(gptconf)
    else:
        model = MimyrModel_kv(gptconf)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(ckp["model"])
    model.eval()
    model.hidden_dim = ckp["model_args"]["n_embd"]
    # model.half()

    tokenizer = MimyrTokenizer(meta_info["token_set"])
    # n_express_level = ckp['model_args']['expression_level']
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    scml = Mimyr(
        adata,
        meta_info,
        tokenizer,
        n_express_level,
        bin_edges=bin_edges,
        model=model,
        **kwargs,
    )

    return scml


def fine_tuning(
    adata: AnnData, meta_info: dict, n_express_level: int = 10, bin_edges=None, **kwargs
):

    tokenizer = MimyrTokenizer(meta_info["token_set"])
    if bin_edges is None:
        bin_edges = compute_global_bin_edges(
            adata, meta_info["gene_set"], n_express_level
        )
    scml = Mimyr(
        adata, meta_info, tokenizer, n_express_level, bin_edges=bin_edges, **kwargs
    )

    return scml


def generate_prompt_for_cg(
    idx: int,
    meta_data: pd.DataFrame,
    meta_pool: dict,
    tokenizer: MimyrTokenizer,
    add_xyz_noise: bool = False,
    min_max_bounds=None,
    randomly_exclude_columns: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Build prompt IDs by scanning meta_pool for keys that are actual
    columns in meta_data.  For each matching key, if the cell's
    value is in meta_pool[key], emit (col_id, val_id).

    Returns
    -------
    prompt_ids : List[int]
        [ col1_id, col2_id, ... ]
    prompt_vals: List[int]
        [ val1_id, val2_id, ... ]
    """
    prompt_ids: List[int] = []
    prompt_vals: List[int] = []

    col_positions = {col: i for i, col in enumerate(meta_data.columns)}

    for col, allowed_vals in meta_pool.items():
        # only consider columns you actually have
        if col not in meta_data.columns:
            continue

        if randomly_exclude_columns and col in randomly_exclude_columns:
            if np.random.rand() < 0.5:
                continue

        col_idx = col_positions[col]
        val = meta_data.iat[idx, col_idx]
        # skip NA or filler
        if pd.isna(val) or str(val) == "Unclassified":
            continue

        # if you’ve specified an allow‐list, enforce it
        if allowed_vals is not None and val not in allowed_vals:
            continue

        # convert column name → token ID
        col_id = tokenizer.convert_tokens_to_ids([str(val)])[0]

        prompt_ids.append(col_id)
        prompt_vals.append(0)

    # for coord in ['<x>','<y>','<z>']:
    #     if coord in meta_data.columns:
    #         col_idx = col_positions[coord]
    #         val = meta_data.iat[idx, col_idx]
    #         prompt_ids.append(tokenizer.convert_tokens_to_ids([coord])[0])
    #         prompt_vals.append(val)

    for coord in ["<x>", "<y>", "<z>"]:
        if coord in meta_data.columns:
            col_idx = col_positions[coord]
            val = meta_data.iat[idx, col_idx]

            if add_xyz_noise:
                val = int(val)  # ensure it's an integer
                noise = np.random.choice([-2, -1, 0, 1, 2])  # discrete noise
                val += noise
                min_val, max_val = min_max_bounds
                val = max(min_val, min(val, max_val))  # clamp

            prompt_ids.append(tokenizer.convert_tokens_to_ids([coord])[0])
            prompt_vals.append(val)

    return prompt_ids, prompt_vals


def compute_global_bin_edges(
    adata: AnnData, Mimyr_gene_set: List, n_express_level: int, include_0s: bool = False
) -> np.ndarray:
    """
    Scan your entire expression matrix to find the global min/max of all
    positive (or optionally zero‐included) values, then build
    self.n_express_level bins.

    Stores the result in self.bin_edges of length (n_bins+1).
    """
    adata_var = set(adata.var_names.tolist())
    adata2 = adata[:, Mimyr_gene_set]  # subsetted AnnData

    X = adata2.X
    if scipy.sparse.issparse(X):
        vals = X.data  # only nonzero values
    else:
        vals = X.ravel()

    if not include_0s:
        vals = vals[vals > 0]

    if len(vals) == 0:
        raise ValueError("No positive expression values found to build edges")

    min_val = vals.min()
    max_val = vals.max()
    edges = np.linspace(min_val, max_val, n_express_level + 1)
    return edges


def tokens_and_vals_to_expression_row(
    var_names: List[str],
    gene_tokens: List[str],
    gene_tokens_int: List[int],
    new_vals: List[float],
    return_series: bool = False,
    *,
    max_violations: int = 5,
    allow_ties: bool = True,
    enforce_range: bool = True,
    token_min: int = 1,
    token_max: int = 2001,
) -> Union[np.ndarray, pd.Series]:
    """
    Convert Mimyr output (gene tokens + values) into an expression vector aligned to var_names.

    Behavior
    --------
    - If <eos> (0) exists:
        * Take the prefix before it.
        * Trim only the minimal noisy TAIL so that there are at most `max_violations`
          non-descending (i.e., rising) steps at the very end. Walk backward.
        * Optionally drop out-of-range tail tokens first (1..2000) if enforce_range=True.
    - If no <eos>:
        * Scan FORWARD from the start and cut once cumulative violations exceed `max_violations`.
          (No “under-200” logic; entirely removed.)
    - Duplicates → averaged; Missing genes → 0.
    """
    n = min(len(gene_tokens), len(gene_tokens_int), len(new_vals))
    if n == 0:
        out = np.zeros(len(var_names), dtype=float)
        return pd.Series(out, index=var_names) if return_series else out

    gene_tokens = gene_tokens[:n]
    gene_tokens_int = gene_tokens_int[:n]
    new_vals = new_vals[:n]

    def _is_in_range(tok: int) -> bool:
        return (token_min <= tok <= token_max) if enforce_range else True

    def _trim_tail_with_patience(prefix_int: List[int]) -> int:
        """
        Backward trim: allow up to `max_violations` rising steps at the very end.
        Also drop out-of-range TAIL tokens first if enforce_range.
        Returns cut index k (keep prefix_int[:k]).
        """
        if not prefix_int:
            return 0

        cut = len(prefix_int)

        # Drop out-of-range tail tokens first
        if enforce_range:
            i = cut - 1
            while i >= 0 and not _is_in_range(prefix_int[i]):
                cut = i
                i -= 1

        # Walk backward; count rises at the end; cut when budget exceeded
        violations = 0
        i = cut - 1
        while i > 0:
            prev_tok = prefix_int[i - 1]
            curr_tok = prefix_int[i]

            if enforce_range and not _is_in_range(prev_tok):
                cut = i
                break

            # Define violation: a rise (ties allowed if allow_ties=True)
            if allow_ties:
                increased = curr_tok > prev_tok
            else:
                increased = curr_tok >= prev_tok

            if increased:
                violations += 1
                if violations > max_violations:
                    cut = i
                    break
            i -= 1

        return max(0, cut)

    if 0 in gene_tokens_int:
        # Case A: <eos> present → tail patience trim on prefix
        eos_idx = gene_tokens_int.index(0)
        prefix_tokens = gene_tokens[:eos_idx]
        prefix_int = gene_tokens_int[:eos_idx]
        prefix_vals = new_vals[:eos_idx]

        good_len = _trim_tail_with_patience(prefix_int)
        gene_tokens = prefix_tokens[:good_len]
        new_vals = prefix_vals[:good_len]

    else:
        # Case B: no <eos> → forward scan; cut when violations exceed budget
        cut_idx = len(gene_tokens_int)
        violations = 0
        for i in range(1, len(gene_tokens_int)):
            prev_tok = gene_tokens_int[i - 1]
            curr_tok = gene_tokens_int[i]

            # Optionally treat out-of-range as immediate cut
            if enforce_range and (
                not _is_in_range(curr_tok) or not _is_in_range(prev_tok)
            ):
                cut_idx = i
                break

            # Violation definition (rise)
            if allow_ties:
                increased = curr_tok > prev_tok
            else:
                increased = curr_tok >= prev_tok

            if increased:
                violations += 1
                if violations > max_violations:
                    cut_idx = i
                    break

        gene_tokens = gene_tokens[:cut_idx]
        new_vals = new_vals[:cut_idx]

    # Nothing left? return zeros
    if len(gene_tokens) == 0:
        expr = np.zeros(len(var_names), dtype=float)
        return pd.Series(expr, index=var_names) if return_series else expr

    # Aggregate duplicates by mean
    agg = {}
    for g, v in zip(gene_tokens, new_vals):
        if g is None:
            continue
        agg.setdefault(g, []).append(float(v))
    avg = {g: (sum(vals) / len(vals)) for g, vals in agg.items()}

    # Build output aligned to var_names
    expr = np.zeros(len(var_names), dtype=float)
    for i, gene in enumerate(var_names):
        expr[i] = avg.get(gene, 0.0)

    return pd.Series(expr, index=var_names) if return_series else expr
