import os
import torch

from data_modes import is_sagittal_mode
from metrics import (
    soft_accuracy,
    soft_correlation,
    soft_f1,
    delauney_colocalization,
    gridized_l1_distance,
    gridized_kl_divergence,
    celltype_abundance_metrics,
    local_composition_entropy,
    composition_pairwise,
)


class Evaluator:
    def __init__(
        self,
        config,
        metadata_dir="model_checkpoints/metadata",
        gene_set_file=None,
    ):
        self.config = config
        if gene_set_file is not None:
            with open(gene_set_file) as f:
                self.gene_set = [line.strip() for line in f if line.strip()]
        else:
            self.gene_set = torch.load(
                os.path.join(metadata_dir, config["meta_info"])
            )["gene_set"]

    def evaluate(self, predicted_adata, target_adata, sample=100, filter_by_gt=False):
        results = {}

        ### Flatten the 3d to 2d
        if is_sagittal_mode(self.config["data_mode"]):
            target_adata.obsm["aligned_spatial"] = target_adata.obsm["aligned_spatial"][
                :, 1:
            ]
            predicted_adata.obsm["spatial"] = predicted_adata.obsm["spatial"][:, 1:]

        else:
            target_adata.obsm["aligned_spatial"] = target_adata.obsm["aligned_spatial"][
                :, :2
            ]
            predicted_adata.obsm["spatial"] = predicted_adata.obsm["spatial"][:, :2]

        if "gridized_l1_distance" in self.config["metrics"]:
            # for k in [5, 10, 20]:
            for r in [0.3, 0.4, 0.5]:
                tvd = gridized_l1_distance(
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obsm["spatial"],
                    radius=r,
                )
                print("gridized_l1_distance @", r, "r:", tvd)
                results[f"gridized_l1_distance@{r} r"] = tvd

        if "gridized_kl_divergence" in self.config["metrics"]:
            for r in [0.3, 0.4, 0.5]:
                kl = gridized_kl_divergence(
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obsm["spatial"],
                    radius=r,
                )
                print("gridized_kl_divergence @", r, "r:", kl)
                results[f"gridized_kl_divergence@{r} r"] = kl

        if "soft_accuracy" in self.config["metrics"]:
            # for k in [5, 10, 20]:
            for r in [0.03, 0.04, 0.05]:
                sa = soft_accuracy(
                    target_adata.obs["token"].to_numpy().tolist(),
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obs["token"].tolist(),
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                )
                print("soft accuracy @", r, ":", sa)
                results[f"soft_accuracy@{r}"] = sa

        # --- tail-sensitive cell-type composition diagnostics (mode-collapse) ---
        if "celltype_abundance" in self.config["metrics"]:
            ab = celltype_abundance_metrics(
                target_adata.obs["token"].to_numpy(),
                predicted_adata.obs["token"].to_numpy(),
            )
            for name, val in ab.items():
                print("celltype_abundance:", name, ":", val)
                results[f"celltype_abundance_{name}"] = val

        if "composition_entropy" in self.config["metrics"]:
            for r in [0.03, 0.04, 0.05]:
                ce = local_composition_entropy(
                    target_adata.obs["token"].to_numpy(),
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obs["token"].to_numpy(),
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                )
                for name, val in ce.items():
                    print("composition_entropy @", r, name, ":", val)
                    results[f"composition_entropy_{name}@{r}"] = val

        if "composition_pairwise" in self.config["metrics"]:
            for r in [0.03, 0.04, 0.05]:
                pw = composition_pairwise(
                    target_adata.obs["token"].to_numpy(),
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obs["token"].to_numpy(),
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                )
                for name in ["rare_frac_pearson", "rare_frac_spearman", "rare_frac_mae"]:
                    print("composition_pairwise @", r, name, ":", pw[name])
                    results[f"composition_pairwise_{name}@{r}"] = pw[name]
                # strata as separate NUMERIC columns (main.py coerces every result to
                # float, so a joined string would crash the write path).
                for qi, (vg, vp) in enumerate(zip(pw["strata_gt"], pw["strata_pred"])):
                    results[f"composition_pairwise_strata_gt_q{qi}@{r}"] = vg
                    results[f"composition_pairwise_strata_pred_q{qi}@{r}"] = vp

        if "delauney_colocalization" in self.config["metrics"]:
            dc = delauney_colocalization(
                target_adata.obs["token"].to_numpy().tolist(),
                target_adata.obsm["aligned_spatial"],
                predicted_adata.obs["token"].tolist(),
                predicted_adata.obsm["spatial"],
            )
            print("delauney colocalization :", dc)
            results[f"delauney_colocalization"] = dc

        if "neighborhood_enrichment" in self.config["metrics"]:
            for k in [5, 10, 20]:
                ne = neighborhood_enrichment(
                    target_adata.obs["token"].to_numpy().tolist(),
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata.obs["token"].tolist(),
                    predicted_adata.obsm["spatial"],
                    k=k,
                )
                print("neighborhood enrichment @", k, ":", ne)
                results[f"neighborhood_enrichment@{k}"] = ne


        if "soft_spearman_correlation" in self.config["metrics"]:

            for r in [0.03, 0.04, 0.05, 0.09, 0.12, 0.15]:
                sc = soft_correlation(
                    target_adata,
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata,
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                    corr_type="spearman",
                    gene_set=self.gene_set,
                    filter_by_gt=filter_by_gt,
                )
                print("soft spearman correlation radius @", r, ":", sc)
                results[f"soft_spearman_correlation_radius@{r}"] = sc

        if "soft_correlation_top_100" in self.config["metrics"]:

            for r in [0.03, 0.04, 0.05, 0.09, 0.12, 0.15]:
                sc = soft_correlation_top_100(
                    target_adata,
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata,
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                    corr_type="spearman",
                    gene_set=self.gene_set,
                )
                print("soft spearman correlation radius @", r, ":", sc)
                results[f"soft_spearman_correlation_radius@{r}"] = sc

        # F1
        if "soft_f1" in self.config["metrics"]:
            for r in [0.03, 0.04, 0.05, 0.09, 0.12, 0.15]:
                sp = soft_f1(
                    target_adata,
                    target_adata.obsm["aligned_spatial"],
                    predicted_adata,
                    predicted_adata.obsm["spatial"],
                    radius=r,
                    sample=sample,
                    gene_set=self.gene_set,
                    filter_by_gt=filter_by_gt,
                )[0]
                print("soft f1 radius @", r, ":", sp)
                results[f"soft_f1_radius@{r}"] = sp

        return results
