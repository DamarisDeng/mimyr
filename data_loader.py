import os
import random
import scanpy as sc
import pandas as pd
import tqdm
import numpy as np
from gene_exp_model import GeneExpModel
from models.generative_transformer.data_util import harmonize_dataset
import torch


class SliceDataLoader:
    def __init__(
        self,
        mode="intra",
        label="subclass",
        cfg=None,
        metadata_dir="/work/pi_f008n64_dartmouth_edu/Mimyr/spencer_code/models/generative_transformer/metadata/",
    ):
        """
        Args:
            mode (str): 'intra' or 'transfer'
            transfer_from (list, optional): List of slice names for training (only for transfer mode)
            transfer_to (list, optional): List of slice names for val/test (only for transfer mode)
        """
        self.mode = mode

        self.label = label

        self.input_dirs = {
            "type1": "/work/magroup/skrieger/tissue_generator/quantized_slices/subclass_z1_d338_0_rotated",
        }

        self.ccf_csv = "data/ccf_coordinates.csv"

        self.metadata_dir = metadata_dir

        self.train_slices = None
        self.val_slices = None
        self.test_slices = None
        self.gene_exp_model = None
        self.density_model = None
        self.cfg = cfg

    def load_intra_slices(self, fast=False, fast_select=None):
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
                sc.read_h5ad(os.path.join(input_dir, fname))
                for i, fname in enumerate(sorted_slices1)
                if i in [26, 27, 28, 29]
            ]
        elif fast_select:
            slices1 = [
                sc.read_h5ad(os.path.join(input_dir, fname))
                for i, fname in enumerate(sorted_slices1)
                if i in [fast_select - 1, fast_select, fast_select + 1]
            ]
        else:
            slices1 = [
                sc.read_h5ad(os.path.join(input_dir, fname)) for fname in sorted_slices1
            ]
        return slices1

    def load_zhuangn_slices(self, n=2, fast_select=None, remove_edges=True):
        print("Fast selecting", fast_select)

        # Load slices2
        input_dir = (
            f"{self.cfg['data_dir']}/Zhuang-ABCA-{n}"
        )
        if remove_edges:
            sorted_slices2 = sorted(
                [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]
            )[2:-2]
        else:
            sorted_slices2 = sorted(
                [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]
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
            sc.read_h5ad(os.path.join(input_dir, fname))
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
            valid_mask = ~slice.obs[["x_ccf", "y_ccf", "z_ccf"]].isna().any(axis=1)
            slices2[i] = slice[valid_mask].copy()

        return slices2

    def load_diseased_slices(self):
        # Load only slices1
        input_dir = f"{self.cfg['data_dir']}/cleaned_versions"
        sorted_slices1 = [
            "Trem2_5xFAD1_cleaned.h5ad",
            "Trem2_5xFAD2_cleaned.h5ad",
            "Trem2_5xFAD3_cleaned.h5ad",
            "Trem2_5xFAD4_cleaned.h5ad",
            "Trem2_5xFAD5_cleaned.h5ad",
        ]
        return [
            sc.read_h5ad(os.path.join(input_dir, fname)) for fname in sorted_slices1
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
        meta_info = torch.load(f'{self.metadata_dir}{self.cfg["meta_info"]}')
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

        return harmonized_train, harmonized_val, harmonized_test, harmonized_ref

    def _set_slice_attributes(
        self, train_slices, val_slices, test_slices, reference_slices
    ):
        """Helper method to set slice attributes."""
        self.train_slices = train_slices
        self.val_slices = val_slices
        self.test_slices = test_slices
        self.reference_slices = reference_slices
        self.fine_tune_train_slices = None
        self.fine_tune_val_slices = None
        self.fine_tune_test_slices = None

    def prepare(self):
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

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Print summary
        print(f"Prepared data:")
        print(f" - Train: {len(self.train_slices)} slices")
        print(f" - Val: {len(self.val_slices)} slices")
        print(f" - Test: {len(self.test_slices)} slices")
        if self.fine_tune_train_slices is not None:
            print(f" - Fine-tune Train: {len(self.fine_tune_train_slices)} slices")
            print(f" - Fine-tune Val: {len(self.fine_tune_val_slices)} slices")
            print(f" - Fine-tune Test: {len(self.fine_tune_test_slices)} slices")
