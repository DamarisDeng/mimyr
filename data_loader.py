import os
import json
import random
import scanpy as sc
import pandas as pd
import tqdm
import numpy as np
from gene_exp_model import GeneExpModel
from models.generative_transformer.data_util import harmonize_dataset
import torch

from loguru import logger as _loguru_logger
_loguru_logger.disable("slaf")


def _read_slice(path: str, backed: bool = False):
    """Read a slice file, auto-detecting .slaf or .h5ad format.

    If the literal path does not exist, tries swapping the extension between
    .slaf and .h5ad, so hardcoded filename lists work regardless of which
    format is present on disk.
    """
    if not os.path.exists(path):
        stem, _ = os.path.splitext(path)
        for ext in (".slaf", ".h5ad"):
            candidate = stem + ext
            if os.path.exists(candidate):
                return _read_slice(candidate, backed=backed)
        raise FileNotFoundError(
            f"No slice file found at {path!r} (tried .slaf and .h5ad)"
        )

    if path.endswith(".slaf"):
        from slaf.integrations.anndata import read_slaf
        import anndata as _ad
        lazy = read_slaf(path)
        if backed:
            return lazy
        X = lazy.X.compute()
        return _ad.AnnData(X=X, obs=lazy.obs.copy(), var=lazy.var.copy())

    return sc.read_h5ad(path, backed="r" if backed else None)


def _is_slice_file(fname: str) -> bool:
    return fname.endswith((".h5ad", ".slaf"))


def _unique_slice_files(names):
    """Deduplicate slice filenames by stem, preferring .slaf over .h5ad."""
    by_stem = {}
    for name in names:
        stem, ext = os.path.splitext(name)
        if stem not in by_stem or ext == ".slaf":
            by_stem[stem] = name
    return sorted(by_stem.values())


def _set_uniform_sampling_prob(adata, col="sampling_prob"):
    """Set uniform per-cell sampling probabilities (dataloader will renormalize)."""
    adata.obs[col] = np.full(adata.n_obs, 1.0, dtype=np.float64)


def _rebalance_sampling_mass(
    adata_train,
    group_col="dataset_type",
    sampling_col="sampling_prob",
    target_fracs=None,
    fill_missing_with=1.0,
    eps=1e-12,
):
    """
    Rescale sampling_prob so total probability mass per group matches target_fracs.
    Relative weights within each group are preserved.
    """
    if sampling_col not in adata_train.obs.columns:
        adata_train.obs[sampling_col] = fill_missing_with

    p = adata_train.obs[sampling_col].astype(float).to_numpy()
    p = np.nan_to_num(p, nan=fill_missing_with, posinf=fill_missing_with, neginf=0.0)
    adata_train.obs[sampling_col] = p

    if target_fracs is None:
        groups = adata_train.obs[group_col].astype("category")
        cats = [c for c in groups.cat.categories if (groups == c).any()]
        target = {c: 1.0 / max(len(cats), 1) for c in cats}
    else:
        target = target_fracs

    df = adata_train.obs[[group_col, sampling_col]].copy()
    cur = df.groupby(group_col)[sampling_col].sum().to_dict()

    scales = {g: t / max(cur.get(g, 0.0), eps) for g, t in target.items()}
    gvals = adata_train.obs[group_col].to_numpy()
    scale_vec = np.vectorize(lambda g: scales.get(g, 1.0))(gvals)
    adata_train.obs[sampling_col] = adata_train.obs[sampling_col].to_numpy() * scale_vec
    print("rebalanced sampling probs")


class SliceDataLoader:
    def __init__(
        self,
        mode="intra",
        label="subclass",
        cfg=None,
        metadata_dir="/work/pi_f008n64_dartmouth_edu/Mimyr/spencer_code/models/generative_transformer/metadata",
        omit_x=False,
    ):
        """
        Args:
            mode (str): data mode — one of rq1, rq2, rq2_v2, rq3, rq3_v2, rq3_v2_d*,
                rq3_v2_cheat, rq4, rq4_rq3, rq4_noref, rq5, base_zhuang1
            label (str): obs column to use as the cell-type label
            cfg (dict): full config dict (from argparse); must include data_dir and,
                for Zhuang-based modes, zhuang_data_dir; for rq5, diseased_data_dir
            metadata_dir (str): directory containing edges_x/y/z.pkl, hierarchy.pkl,
                and the meta_info .pt file
        """
        self.mode = mode
        self.label = label
        self.ccf_csv = "data/ccf_coordinates.csv"
        # Normalize so downstream f-string path joins always work
        self.metadata_dir = metadata_dir.rstrip(os.sep) + os.sep
        # rq1 test/val indices — used in rq3_v2_rq1 to exclude from training
        self._rq1_test_indices = [8, 16, 29, 34, 43]
        self._rq1_val_indices = [6, 28, 44]

        self.train_slices = None
        self.val_slices = None
        self.test_slices = None
        self.gene_exp_model = None
        self.density_model = None
        self.cfg = cfg
        self.omit_x = omit_x
        print("data loader config: ",self.cfg)

    def load_intra_slices(self, fast=False, fast_select=None, select_indices=None):
        # Load only slices1
        input_dir = self.cfg["data_dir"] + "/MERSCOPE"
        sorted_slices1 = [
            "sec_05.h5ad",
            "sec_06.h5ad",
            "sec_08.h5ad",
            "sec_09.h5ad",
            "sec_10.h5ad",
            "sec_11.h5ad",
            "sec_12.h5ad",
            "sec_13.h5ad",
            "sec_14.h5ad",
            "sec_15.h5ad",
            "sec_16.h5ad",
            "sec_17.h5ad",
            "sec_18.h5ad",
            "sec_19.h5ad",
            "sec_24.h5ad",
            "sec_25.h5ad",
            "sec_26.h5ad",
            "sec_27.h5ad",
            "sec_28.h5ad",
            "sec_29.h5ad",
            "sec_30.h5ad",
            "sec_31.h5ad",
            "sec_32.h5ad",
            "sec_33.h5ad",
            "sec_35.h5ad",
            "sec_36.h5ad",
            "sec_37.h5ad",
            "sec_38.h5ad",
            "sec_39.h5ad",
            "sec_40.h5ad",
            "sec_42.h5ad",
            "sec_43.h5ad",
            "sec_44.h5ad",
            "sec_45.h5ad",
            "sec_46.h5ad",
            "sec_47.h5ad",
            "sec_48.h5ad",
            "sec_49.h5ad",
            "sec_50.h5ad",
            "sec_51.h5ad",
            "sec_52.h5ad",
            "sec_54.h5ad",
            "sec_55.h5ad",
            "sec_56.h5ad",
            "sec_57.h5ad",
            "sec_58.h5ad",
            "sec_59.h5ad",
            "sec_60.h5ad",
            "sec_61.h5ad",
            "sec_62.h5ad",
            "sec_64.h5ad",
            "sec_66.h5ad",
            "sec_67.h5ad",
        ]
        if fast:
            slices1 = [
                _read_slice(os.path.join(input_dir, fname))
                for i, fname in enumerate(sorted_slices1)
                if i in [26, 27, 28, 29]
            ]
        elif fast_select:
            slices1 = [
                _read_slice(os.path.join(input_dir, fname))
                for i, fname in enumerate(sorted_slices1)
                if i in [fast_select - 1, fast_select, fast_select + 1]
            ]
        elif select_indices is not None:
            idx_set = set(select_indices)
            slices1 = [
                _read_slice(os.path.join(input_dir, fname))
                for i, fname in enumerate(sorted_slices1)
                if i in idx_set
            ]
        else:
            slices1 = [
                _read_slice(os.path.join(input_dir, fname)) for fname in sorted_slices1
            ]
        for s in slices1:
            s.obs["technology"] = "M550"
            s.obs['disease_state'] = 'healthy'
            s.obs['species'] = 'mouse'
        return slices1

    def _compute_rq1_train_indices_by_z_ccf(self, test_slices, rq1_eligible_indices):
        """For each test slice, find the eligible rq1 slice with the closest mean z_ccf.

        Reads eligible rq1 h5ad files in backed mode (obs only) to avoid loading the
        full expression matrix.

        Args:
            test_slices: list of AnnData objects with obs["z_ccf"] already populated
            rq1_eligible_indices: list of candidate rq1 positional indices

        Returns:
            sorted, deduplicated list of selected rq1 indices
        """
        input_dir = os.path.join(self.cfg["data_dir"], "MERSCOPE")
        sorted_slices1 = [
            "sec_05.h5ad", "sec_06.h5ad", "sec_08.h5ad", "sec_09.h5ad",
            "sec_10.h5ad", "sec_11.h5ad", "sec_12.h5ad", "sec_13.h5ad",
            "sec_14.h5ad", "sec_15.h5ad", "sec_16.h5ad", "sec_17.h5ad",
            "sec_18.h5ad", "sec_19.h5ad", "sec_24.h5ad", "sec_25.h5ad",
            "sec_26.h5ad", "sec_27.h5ad", "sec_28.h5ad", "sec_29.h5ad",
            "sec_30.h5ad", "sec_31.h5ad", "sec_32.h5ad", "sec_33.h5ad",
            "sec_35.h5ad", "sec_36.h5ad", "sec_37.h5ad", "sec_38.h5ad",
            "sec_39.h5ad", "sec_40.h5ad", "sec_42.h5ad", "sec_43.h5ad",
            "sec_44.h5ad", "sec_45.h5ad", "sec_46.h5ad", "sec_47.h5ad",
            "sec_48.h5ad", "sec_49.h5ad", "sec_50.h5ad", "sec_51.h5ad",
            "sec_52.h5ad", "sec_54.h5ad", "sec_55.h5ad", "sec_56.h5ad",
            "sec_57.h5ad", "sec_58.h5ad", "sec_59.h5ad", "sec_60.h5ad",
            "sec_61.h5ad", "sec_62.h5ad", "sec_64.h5ad", "sec_66.h5ad",
            "sec_67.h5ad",
        ]

        print("rq3_v2_rq1: computing rq1 mean z_ccf for eligible slices...")
        rq1_z_means = {}
        rq1_y_means = {}
        rq1_x_means = {}
        for i in rq1_eligible_indices:
            fname = sorted_slices1[i]
            s = _read_slice(os.path.join(input_dir, fname), backed=True)
            rq1_z_means[i] = float(s.obs["z_ccf"].mean())
            rq1_y_means[i] = float(s.obs["y_ccf"].mean())
            rq1_x_means[i] = float(s.obs["x_ccf"].mean())
            if hasattr(s, "file"):
                s.file.close()
        print(rq1_z_means)
        print(rq1_y_means)
        print(rq1_x_means)

        selected = []
        for ts in test_slices:
            test_z = float(ts.obs["z_ccf"].mean())
            test_y = float(ts.obs["y_ccf"].mean())
            test_x = float(ts.obs["x_ccf"].mean())
            nearest = min(rq1_eligible_indices, key=lambda i: abs(rq1_x_means[i] - test_x))
            if nearest not in selected:
                selected.append(nearest)
            print(
                f"  test z_ccf={test_z:.4f} → rq1[{nearest}] "
                f"  test y_ccf={test_y:.4f}, test x_ccf={test_x:.4f}"
                f"({sorted_slices1[nearest]}) z_ccf={rq1_x_means[nearest]:.4f}"
            )

        return sorted(selected)

    def load_zhuangn_slices(self, n=2, fast_select=None, remove_edges=True):
        print("Fast selecting", fast_select)

        zhuang_base = self.cfg.get("zhuang_data_dir")
        if not zhuang_base:
            print(self.cfg)
            raise ValueError(
                f"data mode '{self.mode}' requires --zhuang_data_dir (or zhuang_data_dir in config.yaml) "
                f"to be set to the directory containing Zhuang-ABCA-* subdirectories."
            )
        input_dir = os.path.join(zhuang_base, f"Zhuang-ABCA-{n}")
        if remove_edges:
            sorted_slices2 = _unique_slice_files(
                [f for f in os.listdir(input_dir) if _is_slice_file(f)]
            )[2:-2]
        else:
            sorted_slices2 = _unique_slice_files(
                [f for f in os.listdir(input_dir) if _is_slice_file(f)]
            )
        print(sorted_slices2)
        if fast_select:
            sorted_slices2 = [
                f
                for i, f in enumerate(sorted_slices2)
                if i in [fast_select - 1, fast_select, fast_select + 1]
            ]
            print("fs")
        else:
            sorted_slices2 = [f for i, f in enumerate(sorted_slices2)]
        print(sorted_slices2)
        slices2 = [
            _read_slice(os.path.join(input_dir, fname))
            for fname in tqdm.tqdm(sorted_slices2)
        ]

        # CCF coordinate filtering
        df_ccf = pd.read_csv(
            f"{self.cfg['data_dir']}/ccf-ABCA-{n}.csv"
        ).set_index("cell_label")
        for i, slice in enumerate(slices2):
            df_filtered = df_ccf.loc[df_ccf.index.intersection(slice.obs_names)]
            df_filtered = df_filtered.reindex(slice.obs_names)
            slice.obs["x_ccf"] = df_filtered["x"]
            slice.obs["y_ccf"] = df_filtered["y"]
            slice.obs["z_ccf"] = df_filtered["z"]
            slice.obs["technology"] = "M1100"
            slice.obs['disease_state'] = 'healthy'
            slice.obs['species'] = 'mouse'
            valid_mask = ~slice.obs[["x_ccf", "y_ccf", "z_ccf"]].isna().any(axis=1)
            slices2[i] = slice[valid_mask].copy()

        slices2 = [s for s in slices2 if s.n_obs > 0]
        print(f'Length of slices2 is {len(slices2)}')
        return slices2

    def load_diseased_slices(self):
        input_dir = self.cfg.get("diseased_data_dir")
        if not input_dir:
            raise ValueError(
                "data mode 'rq5' requires --diseased_data_dir (or diseased_data_dir in config.yaml) "
                "to be set to the directory containing the 5xFAD Trem2 .h5ad files."
            )
        sorted_slices1 = [
            "Trem2_5xFAD1_cleaned.h5ad",
            "Trem2_5xFAD2_cleaned.h5ad",
            "Trem2_5xFAD3_cleaned.h5ad",
            "Trem2_5xFAD4_cleaned.h5ad",
            "Trem2_5xFAD5_cleaned.h5ad",
        ]
        return [
            _read_slice(os.path.join(input_dir, fname)) for fname in sorted_slices1
        ]


    def _align_and_tokenize_slices(self, slices):
        """Helper method to align and tokenize slices."""
        for slice in slices:
            slice.obsm["aligned_spatial"] = np.stack(
                [slice.obs["z_ccf"], slice.obs["y_ccf"], slice.obs["x_ccf"]], -1
            )

        self.gene_exp_model = GeneExpModel(slices, label=self.label)
        self.gene_exp_model.fit()
        return self.gene_exp_model.get_tokenized_slices()

    def _harmonize_slice_lists(
        self,
        train_slices,
        val_slices,
        test_slices,
        reference_slices,
        technology=None,
        overwrite_technology=False,
    ):
        """Helper method to harmonize all slice lists with metadata."""
        meta_info = torch.load(os.path.join(self.metadata_dir, self.cfg["meta_info"]))
        edges = [
            f"{self.metadata_dir}edges_x.pkl",
            f"{self.metadata_dir}edges_y.pkl",
            f"{self.metadata_dir}edges_z.pkl",
        ]

        harmonize_kwargs = {}
        if technology is not None:
            harmonize_kwargs["technology"] = technology
        if overwrite_technology:
            harmonize_kwargs["overwrite_technology"] = overwrite_technology

        harmonized_train = [
            harmonize_dataset(s, meta_info, edges, **harmonize_kwargs)
            for s in train_slices
        ]
        harmonized_val = [
            harmonize_dataset(s, meta_info, edges, **harmonize_kwargs)
            for s in val_slices
        ]
        harmonized_test = [
            harmonize_dataset(s, meta_info, edges, **harmonize_kwargs)
            for s in test_slices
        ]
        harmonized_ref = [
            harmonize_dataset(s, meta_info, edges, **harmonize_kwargs)
            for s in reference_slices
        ]

        if self.omit_x:
            for s in harmonized_train + harmonized_val + harmonized_test + harmonized_ref:
                if "<x>" in s.obs.columns:
                    del s.obs["<x>"]

        return harmonized_train, harmonized_val, harmonized_test, harmonized_ref

    def _set_slice_attributes(
        self, train_slices, val_slices, test_slices, reference_slices
    ):
        """Helper method to set slice attributes."""
        self.train_slices = train_slices
        self.val_slices = val_slices
        self.test_slices = test_slices
        self.reference_slices = reference_slices
        # Most modes pair each test slice with exactly two bracketing reference
        # slices, consumed positionally as reference_slices[2*i:2*i+2] in main.py.
        # Modes that instead expose one whole-brain reference pool shared by every
        # test slice set this True; main.py then hands the full list through.
        self.fixed_reference_pool = False
        self.fine_tune_train_slices = None
        self.fine_tune_val_slices = None
        self.fine_tune_test_slices = None

    def _build_adata(
        self,
        adata2_path=None,
        gene_set=None,
        seed=0,
        dummy=False,
        sampling_col="sampling_prob",
        output_dir=None,
        rank=0,
    ):
        """
        Concatenate train/val/test slices into self.adata_train / self.adata_val / self.adata_test.
        If adata2_path is provided, load and merge it into the training set.
        """
        adata_train = (
            sc.concat(self.train_slices, join="outer")
            if len(self.train_slices) > 1
            else self.train_slices[0].copy()
        )
        adata_train.obs_names_make_unique()

        adata_val = (
            sc.concat(self.val_slices, join="outer")
            if len(self.val_slices) > 1
            else self.val_slices[0].copy()
        )
        adata_val.obs_names_make_unique()

        if self.test_slices:
            adata_test = (
                sc.concat(self.test_slices, join="outer")
                if len(self.test_slices) > 1
                else self.test_slices[0].copy()
            )
            adata_test.obs_names_make_unique()
        else:
            adata_test = None

        if adata2_path is not None:
            if output_dir is not None:
                train_stem = os.path.join(output_dir, "train")
                train_path = next(
                    (train_stem + ext for ext in (".slaf", ".h5ad") if os.path.exists(train_stem + ext)),
                    None,
                )
                if train_path is not None:
                    print(f"Found existing {train_path}, loading instead of rebuilding...")
                    self.adata_train = _read_slice(train_path)
                    self.adata_val = adata_val
                    self.adata_test = adata_test
                    return

            rng = np.random.default_rng(seed=seed)
            adata2 = _read_slice(adata2_path)
            adata2.var_names_make_unique()
            if gene_set is not None:
                adata2._inplace_subset_var(gene_set)
            if dummy:
                adata2._inplace_subset_obs(adata2.obs_names[:100])
                print("using only first 100 cells from adata2")
            print(adata2)

            n_val_req = adata_val.n_obs
            n_train_req = adata_train.n_obs
            n_total = adata2.n_obs

            n_val = min(n_val_req, n_total)
            n_train = min(n_train_req, n_total - n_val)

            all_idx = np.arange(n_total)
            idx_val = (
                rng.choice(all_idx, size=n_val, replace=False)
                if n_val > 0
                else np.array([], dtype=int)
            )
            mask_val = np.zeros(n_total, dtype=bool)
            mask_val[idx_val] = True
            remain = np.flatnonzero(~mask_val)
            idx_train = (
                rng.choice(remain, size=n_train, replace=False)
                if n_train > 0
                else np.array([], dtype=int)
            )

            adata2_val = adata2[idx_val].copy()
            adata2._inplace_subset_obs(idx_train)
            adata2_train = adata2

            adata2_train.obs_names_make_unique()
            adata2_val.obs_names_make_unique()
            adata_train.obs_names_make_unique()

            print(
                f"Requested: train={n_train_req}, val={n_val_req}; "
                f"Allocated: train={n_train}, val={n_val} (total available={n_total})"
            )

            _set_uniform_sampling_prob(adata2_train, col=sampling_col)

            adata_train = sc.concat(
                [adata_train, adata2_train],
                join="outer",
                label="dataset_type",
                keys=["st", "scrna"],
            )

            _rebalance_sampling_mass(
                adata_train,
                group_col="dataset_type",
                sampling_col=sampling_col,
                target_fracs={"st": 0.7, "scrna": 0.3},
            )
            print(adata_train)

            if output_dir is not None and rank == 0:
                from slaf.data.converter import SLAFConverter
                SLAFConverter(chunked=False).convert_anndata(adata_train, os.path.join(output_dir, "train.slaf"))

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.adata_test = adata_test
        # Free per-slice objects now that they are concatenated
        self.train_slices = None
        self.val_slices = None

    def prepare(
        self,
        adata2_path=None,
        gene_set=None,
        seed=0,
        dummy=False,
        sampling_col="sampling_prob",
        output_dir=None,
        rank=0,
    ):
        if self.mode == "rq1":
            slices = self.load_intra_slices()
            slices_tokenized = self._align_and_tokenize_slices(slices)
            for sid, slice in enumerate(slices_tokenized):
                slice.obs["slice_id"] = sid

            test_indices = [8, 16, 29, 34, 43]
            val_indices = [6, 28, 44, 14]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [
                i
                for i in range(len(slices_tokenized))
                if i not in test_indices and i not in val_indices
            ]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = []
            for i in test_indices:
                if i - 1 >= 0:
                    ref_indices.append(i - 1)
                if i + 1 < len(slices_tokenized):
                    ref_indices.append(i + 1)
            ref_indices = sorted(set(ref_indices))
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq2":
            slices = self.load_zhuangn_slices(fast_select=33, n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            hole_seeds = [3885, 4632, 9765, 10192, 27389]

            test_slices = []
            masks = []
            self.hole_centers = []
            for hs in hole_seeds:
                print(hs)
                new_slice = slices_tokenized[1].copy()
                mask = (
                    np.linalg.norm(
                        new_slice.obsm["aligned_spatial"]
                        - new_slice.obsm["aligned_spatial"][hs],
                        axis=1,
                    )
                    < 0.3
                )
                new_slice = new_slice[mask]
                masks.append(mask)
                test_slices.append(new_slice)
                self.hole_centers.append(new_slice.obsm["aligned_spatial"].mean(0))

            logical_or_mask = np.logical_or.reduce(masks)
            remaining_cells_slice = slices_tokenized[1][~logical_or_mask]

            train_mask = np.random.rand(len(remaining_cells_slice)) < 0.9
            train_slices = [remaining_cells_slice[train_mask]]
            val_slices = [remaining_cells_slice[~train_mask]]

            reference_slices = [remaining_cells_slice for t in test_slices] + [
                remaining_cells_slice for t in test_slices
            ]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices,
                    val_slices,
                    test_slices,
                    reference_slices,
                    technology="scRNA-seq",
                    overwrite_technology=True,
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq2_v2":
            slices = self.load_zhuangn_slices(fast_select=33, n=2)
            rq1_ref_slices = self.load_intra_slices()

            slices_tokenized = self._align_and_tokenize_slices(slices)
            slices_tokenized_rq1_ref = self._align_and_tokenize_slices(rq1_ref_slices)

            hole_seeds = [3885, 4632, 9765, 10192, 27389]

            test_slices = []
            masks = []
            self.hole_centers = []
            for hs in hole_seeds:
                print(hs)
                new_slice = slices_tokenized[1].copy()
                mask = (
                    np.linalg.norm(
                        new_slice.obsm["aligned_spatial"]
                        - new_slice.obsm["aligned_spatial"][hs],
                        axis=1,
                    )
                    < 0.3
                )
                new_slice = new_slice[mask]
                masks.append(mask)
                test_slices.append(new_slice)
                self.hole_centers.append(new_slice.obsm["aligned_spatial"].mean(0))

            logical_or_mask = np.logical_or.reduce(masks)
            remaining_cells_slice = slices_tokenized[1][~logical_or_mask]

            train_mask = np.random.rand(len(remaining_cells_slice)) < 0.9
            train_slices = [remaining_cells_slice[train_mask]]
            val_slices = [remaining_cells_slice[~train_mask]]

            reference_slices = [
                slices_tokenized_rq1_ref[25],
                slices_tokenized_rq1_ref[26],
                slices_tokenized_rq1_ref[25],
                slices_tokenized_rq1_ref[26],
                slices_tokenized_rq1_ref[25],
                slices_tokenized_rq1_ref[26],
                slices_tokenized_rq1_ref[25],
                slices_tokenized_rq1_ref[26],
                slices_tokenized_rq1_ref[25],
                slices_tokenized_rq1_ref[26],
            ]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices,
                    val_slices,
                    test_slices,
                    reference_slices,
                    technology="scRNA-seq",
                    overwrite_technology=True,
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq3":
            slices = self.load_zhuangn_slices(n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [18, 24, 32, 40, 44]
            val_indices = [29]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [15, 21, 37, 42]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = [15, 21, 21, 29, 29, 37, 37, 42, 42, 42]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq3_v2":
            slices = self.load_zhuangn_slices(n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [1, 10, 20, 30, 40]
            val_indices = [44]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [5, 15, 25, 35]
            ref_indices = [5, 5, 5, 15, 15, 25, 25, 35, 35, 44]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices = [slices_tokenized[i] for i in train_indices]

            if self.cfg.get("use_rq1_train"):
                # Supplement Zhuang training slices with rq1 slices whose mean x_ccf
                # most closely matches each test slice.
                n_rq1_slices = 53
                rq1_train_eligible = list(range(n_rq1_slices))
                test_key = "_".join(str(i) for i in sorted(test_indices))
                cache_path = os.path.join(
                    self.metadata_dir, f"train_indices_rq1_{test_key}.json"
                )
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        train_indices_rq1 = json.load(f)
                    print(
                        f"rq3_v2+rq1: loaded rq1 train indices from cache "
                        f"({cache_path}): {train_indices_rq1}"
                    )
                else:
                    test_slices_for_matching = [slices_tokenized[i] for i in test_indices]
                    train_indices_rq1 = self._compute_rq1_train_indices_by_z_ccf(
                        test_slices_for_matching, rq1_train_eligible
                    )
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(train_indices_rq1, f)
                    print(
                        f"rq3_v2+rq1: computed rq1 train indices by x_ccf and saved to "
                        f"{cache_path}: {train_indices_rq1}"
                    )
                rq1_slices_raw = self.load_intra_slices(select_indices=train_indices_rq1)
                rq1_slices_tokenized = self._align_and_tokenize_slices(rq1_slices_raw)
                train_slices.extend(rq1_slices_tokenized)

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq3_v2_cheat":
            slices = self.load_zhuangn_slices(n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [1, 10, 20, 30, 40]
            val_indices = [44]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [1, 10, 20, 30, 40]
            ref_indices = [5, 5, 5, 15, 15, 25, 25, 35, 35, 44]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices = [slices_tokenized[i] for i in train_indices]

            if self.cfg.get("use_rq1_train"):
                # Supplement Zhuang training slices with rq1 slices whose mean x_ccf
                # most closely matches each test slice.
                n_rq1_slices = 53
                rq1_train_eligible = list(range(n_rq1_slices))
                test_key = "_".join(str(i) for i in sorted(test_indices))
                cache_path = os.path.join(
                    self.metadata_dir, f"train_indices_rq1_{test_key}.json"
                )
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        train_indices_rq1 = json.load(f)
                    print(
                        f"rq3_v2+rq1: loaded rq1 train indices from cache "
                        f"({cache_path}): {train_indices_rq1}"
                    )
                else:
                    test_slices_for_matching = [slices_tokenized[i] for i in test_indices]
                    train_indices_rq1 = self._compute_rq1_train_indices_by_z_ccf(
                        test_slices_for_matching, rq1_train_eligible
                    )
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(train_indices_rq1, f)
                    print(
                        f"rq3_v2+rq1: computed rq1 train indices by x_ccf and saved to "
                        f"{cache_path}: {train_indices_rq1}"
                    )
                rq1_slices_raw = self.load_intra_slices(select_indices=train_indices_rq1)
                rq1_slices_tokenized = self._align_and_tokenize_slices(rq1_slices_raw)
                train_slices.extend(rq1_slices_tokenized)

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode in ("rq3_v2_d4", "rq3_v2_d3", "rq3_v2_d2", "rq3_v2_d1"):
            d = int(self.mode[-1])
            slices = self.load_zhuangn_slices(n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [1, 10, 20, 30, 40]

            test_slices = [slices_tokenized[i] for i in test_indices]

            # Each test slice T gets a symmetric reference bracket [T-d, T+d]
            # (one slice below, one above), flattened in test-slice order so the
            # positional pairing in main.py (reference_slices[2*i:2*i+2]) hands
            # each test slice its below/above pair. The lower index is clamped to
            # 0 for the edge test slice (T=1), which has no slice far enough
            # below it. This is the d5 structure of rq3_v2 generalized to d<5.
            def _pair(T):
                return [max(0, T - d), T + d]

            ref_indices = (
                _pair(1) + _pair(10) + _pair(20) + _pair(30) + _pair(40)
            )

            # Hold out the topmost reference slice as validation (mirrors rq3_v2,
            # where the top reference index doubles as the val slice), and train
            # on every other distinct reference slice.
            val_index = 40 + d
            val_indices = [val_index]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = []
            _seen = set()
            for _idx in ref_indices:
                if _idx == val_index or _idx in _seen:
                    continue
                _seen.add(_idx)
                train_indices.append(_idx)

            train_slices = [slices_tokenized[i] for i in train_indices]

            if self.cfg.get("use_rq1_train"):
                # Supplement Zhuang training slices with rq1 slices whose mean x_ccf
                # most closely matches each test slice.
                n_rq1_slices = 53
                rq1_train_eligible = list(range(n_rq1_slices))
                test_key = "_".join(str(i) for i in sorted(test_indices))
                cache_path = os.path.join(
                    self.metadata_dir, f"train_indices_rq1_{test_key}.json"
                )
                if os.path.exists(cache_path):
                    with open(cache_path) as f:
                        train_indices_rq1 = json.load(f)
                    print(
                        f"rq3_v2_d{d}+rq1: loaded rq1 train indices from cache "
                        f"({cache_path}): {train_indices_rq1}"
                    )
                else:
                    test_slices_for_matching = [slices_tokenized[i] for i in test_indices]
                    train_indices_rq1 = self._compute_rq1_train_indices_by_z_ccf(
                        test_slices_for_matching, rq1_train_eligible
                    )
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(train_indices_rq1, f)
                    print(
                        f"rq3_v2_d{d}+rq1: computed rq1 train indices by x_ccf and saved to "
                        f"{cache_path}: {train_indices_rq1}"
                    )
                
                rq1_slices_raw = self.load_intra_slices(select_indices=train_indices_rq1)
                rq1_slices_tokenized = self._align_and_tokenize_slices(rq1_slices_raw)
                train_slices.extend(rq1_slices_tokenized)


            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq3_v2_d1_cellval":
            slices = self.load_zhuangn_slices(n=2)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [1, 10, 20, 30, 40]
            pool_indices = [9, 19, 29, 39, 44]

            test_slices = [slices_tokenized[i] for i in test_indices]

            pool_combined = sc.concat(
                [slices_tokenized[i] for i in pool_indices], join="outer"
            )
            pool_combined.obs_names_make_unique()
            train_mask = np.random.rand(pool_combined.n_obs) < 0.9
            train_slices = [pool_combined[train_mask].copy()]
            val_slices = [pool_combined[~train_mask].copy()]

            ref_indices = [9] * 3 + [19] * 2 + [29] * 2 + [39] * 2 + [44]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq4_rq3":
            slices = self.load_zhuangn_slices(n=3, remove_edges=False)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [2, 5, 11, 14, 18]
            val_indices = [7]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [0, 3, 12, 17]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = [0, 3, 3, 7, 7, 12, 12, 17, 17, 17]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            slices_rq3 = self.load_zhuangn_slices(n=2)
            slices_tokenized_rq3 = self._align_and_tokenize_slices(slices_rq3)

            train_slices.extend(slices_tokenized_rq3)

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq4":
            slices = self.load_zhuangn_slices(n=3, remove_edges=False)
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [2, 5, 11, 14, 18]
            val_indices = [7]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [0, 3, 12, 17]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = [0, 3, 3, 7, 7, 12, 12, 17, 17, 17]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "rq4_noref":
            # "No-reference" rq4: same held-out Zhuang-ABCA-3 sagittal test sections
            # as rq4, but the weak-teacher-forcing references come from a *different
            # brain*, Zhuang-ABCA-1, which was sectioned coronally. Nothing from the
            # rq4 brain is used at inference time.
            #
            # The orientation difference needs no special handling: both datasets
            # carry Allen CCF coordinates, and aligned_spatial is stacked as
            # [z_ccf, y_ccf, x_ccf] for both (see _align_and_tokenize_slices). Taken
            # together the 129 coronal Zhuang-1 sections are a dense 3-D point cloud
            # of a whole brain (~0.09 mm section spacing in x_ccf, dense in-plane),
            # so a sagittal query plane simply cuts through it and the existing 3-D
            # per-cell-type KDTree lookup in Inference._build_lookup_prior finds the
            # geometrically nearest same-type cell. The rq4 test sections sit at
            # z_ccf ~ 5.11 / 4.54 / 3.26 / 2.51 / 1.62, all inside Zhuang-1's z_ccf
            # coverage [0.51, 5.87].
            #
            # Tokenize the Zhuang-1 pool first so self.gene_exp_model ends up holding
            # the Zhuang-3 model, matching plain rq4 (inference.py reads it for
            # get_label_from_token). Cross-dataset token agreement relies on
            # GeneExpModel loading the global data/id_to_subclass.pkl -- its except
            # branch silently re-enumerates, which would make the two datasets'
            # tokens disagree, so assert the maps match.
            reference_slices = self._align_and_tokenize_slices(
                self.load_zhuangn_slices(n=1)
            )
            ref_token_map = dict(self.gene_exp_model.subclass_to_id)

            slices = self.load_zhuangn_slices(n=3, remove_edges=False)
            slices_tokenized = self._align_and_tokenize_slices(slices)
            if self.gene_exp_model.subclass_to_id != ref_token_map:
                raise RuntimeError(
                    "rq4_noref: cell-type token maps diverged between Zhuang-1 and "
                    "Zhuang-3 -- data/id_to_subclass.pkl failed to load, so tokens "
                    "were re-enumerated per dataset and the cross-brain lookup would "
                    "match the wrong cell types."
                )

            # Same held-out sections as rq4. train/val are unused at inference but
            # _build_adata indexes train_slices[0], so they must stay non-empty.
            test_indices = [2, 5, 11, 14, 18]
            val_indices = [7]
            train_indices = [0, 3, 12, 17]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]
            train_slices = [slices_tokenized[i] for i in train_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )
            # Whole-brain pool shared by every test slice, not a per-slice bracket.
            self.fixed_reference_pool = True

        elif self.mode == "rq5":
            slices = self.load_diseased_slices()
            slices_tokenized = self._align_and_tokenize_slices(slices)

            test_indices = [1]
            val_indices = [0]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [2, 3, 4]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = [0, 3]
            reference_slices = [slices_tokenized[i] for i in ref_indices]

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        elif self.mode == "base_zhuang1":
            slices = self.load_zhuangn_slices(n=1)
            slices_tokenized = self._align_and_tokenize_slices(slices)
            del slices
            for sid, s in enumerate(slices_tokenized):
                s.obs["slice_id"] = sid

            test_indices = [14, 43, 71, 100, 128]
            val_indices = [28, 57, 85, 114]

            test_slices = [slices_tokenized[i] for i in test_indices]
            val_slices = [slices_tokenized[i] for i in val_indices]

            train_indices = [
                i
                for i in range(len(slices_tokenized))
                if i not in test_indices and i not in val_indices
            ]
            train_slices = [slices_tokenized[i] for i in train_indices]

            ref_indices = []
            for i in test_indices:
                if i - 1 >= 0:
                    ref_indices.append(i - 1)
                if i + 1 < len(slices_tokenized):
                    ref_indices.append(i + 1)
            ref_indices = sorted(set(ref_indices))
            reference_slices = [slices_tokenized[i] for i in ref_indices]
            del slices_tokenized

            train_slices, val_slices, test_slices, reference_slices = (
                self._harmonize_slice_lists(
                    train_slices, val_slices, test_slices, reference_slices
                )
            )
            self._set_slice_attributes(
                train_slices, val_slices, test_slices, reference_slices
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if dummy:
            self.train_slices = [s[s.obs_names[:500]] for s in self.train_slices]
            self.val_slices = [s[s.obs_names[:100]] for s in self.val_slices]

        self._build_adata(
            adata2_path=adata2_path,
            gene_set=gene_set,
            seed=seed,
            dummy=dummy,
            sampling_col=sampling_col,
            output_dir=output_dir,
            rank=rank,
        )

        # Print summary
        print(f"Prepared data:")
        print(f" - Train: {len(self.train_slices) if self.train_slices is not None else 'N/A (freed)'} slices")
        print(f" - Val: {len(self.val_slices) if self.val_slices is not None else 'N/A (freed)'} slices")
        print(f" - Test: {len(self.test_slices)} slices")
        if self.fine_tune_train_slices is not None:
            print(f" - Fine-tune Train: {len(self.fine_tune_train_slices)} slices")
            print(f" - Fine-tune Val: {len(self.fine_tune_val_slices)} slices")
            print(f" - Fine-tune Test: {len(self.fine_tune_test_slices)} slices")
