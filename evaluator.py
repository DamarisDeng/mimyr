import torch

from metrics import (
    soft_accuracy,
    soft_correlation,
    soft_f1,
    delauney_colocalization,
    gridized_l1_distance,
)


class Evaluator:
    def __init__(
        self,
        config,
        metadata_dir="/work/pi_f008n64_dartmouth_edu/Mimyr/spencer_code/models/generative_transformer/metadata/",
    ):
        self.config = config
        self.gene_set = torch.load(f"{metadata_dir}{config['meta_info']}")["gene_set"]

    def evaluate(self, predicted_adata, target_adata, sample=100):
        results = {}

        ### Flatten the 3d to 2d
        if self.config["data_mode"] in ["rq4"]:
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
                )[0]
                print("soft f1 radius @", r, ":", sp)
                results[f"soft_f1_radius@{r}"] = sp

        return results
