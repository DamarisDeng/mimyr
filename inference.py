import pickle as pkl
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from metrics import *
from analysis import *

# after the star imports so neither can shadow these
from data_modes import is_sagittal_mode, section_axis

from models.generative_transformer.Mimyr import (
    compute_global_bin_edges,
    generate_prompt_for_cg,
    Mimyr,
    model_generate,
    model_inference,
)


class Inference:
    def __init__(self, combined_model, location_model, slice_data_loader, config):
        self.slice_data_loader = slice_data_loader
        self.token_mapping_model = slice_data_loader.gene_exp_model
        self.location_model = (combined_model.trainer, location_model)
        self.subclass_model = combined_model.celltype_model
        self.config = config

    # === LOCATION ===
    def infer_location(self, adata, new_tissue):
        loc_type = self.config["location_inference_type"]

        if loc_type == "model":

            if self.config["data_mode"] == "rq2":
                xyz = self.location_model[0].sample_with_guidance(
                    self.slice_data_loader.train_slices[0].n_obs * 3,
                    self.location_model[1],
                    guidance_scale=self.config["guidance_signal"],
                    use_ema=False,
                    cond_vec=np.array(
                        [
                            [
                                4,
                                5,
                                new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                                0.0,
                                0.0,
                                1.0,
                            ]
                        ]
                    ),
                )
                new_column = np.full(
                    (len(xyz), 1), new_tissue.obsm["aligned_spatial"].mean(0)[-1]
                )
                xyz = np.concatenate([xyz[:, :2], new_column], axis=1)
                xyz = xyz[
                    np.linalg.norm(xyz - self.slice_data_loader.hole_centers[0], axis=1)
                    < 0.3
                ][: len(new_tissue)]

            elif self.config["data_mode"] == "rq3":
                xyz = self.location_model[0].sample_with_guidance(
                    int(adata.n_obs * 2.5),
                    self.location_model[1],
                    guidance_scale=self.config["guidance_signal"],
                    use_ema=False,
                    cond_vec=np.array(
                        [
                            [
                                4,
                                5,
                                new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                                0.0,
                                0.0,
                                1.0,
                            ]
                        ]
                    ),
                )
                new_column = np.full(
                    (len(xyz), 1), new_tissue.obsm["aligned_spatial"].mean(0)[-1]
                )
                xyz = np.concatenate([xyz[:, :2], new_column], axis=1)
                xyz = xyz[
                    (xyz[:, 1] > new_tissue.obsm["aligned_spatial"].min(0)[1])
                    & (xyz[:, 1] < new_tissue.obsm["aligned_spatial"].max(0)[1])
                    & (xyz[:, 0] > new_tissue.obsm["aligned_spatial"].min(0)[0])
                    & (xyz[:, 0] < new_tissue.obsm["aligned_spatial"].max(0)[0])
                ]

            elif is_sagittal_mode(self.config["data_mode"]):
                # NOTE: for rq4 main.py builds the KDE from a 3-D slice
                # (aligned_spatial keeps all three columns), while the diffusion
                # sampler works in 2-D -- so a *guided* rq4 run would hit a
                # dimension mismatch in potential_model(). Unguided rq4
                # (guidance_signal=0, the default) short-circuits to sample()
                # and is unaffected. Left as-is: no rq4 guided run to test against.
                xyz = self.location_model[0].sample_with_guidance(
                    adata.n_obs * 2,
                    self.location_model[1],
                    guidance_scale=self.config["guidance_signal"],
                    use_ema=False,
                    cond_vec=np.array(
                        [
                            [
                                new_tissue.obsm["aligned_spatial"].mean(0)[0],
                                5,
                                5,
                                1.0,
                                0.0,
                                0.0,
                            ]
                        ]
                    ),
                )
                ## Add new_tissue.obsm["aligned_spatial"].mean(0)[0] as a new column
                new_column = np.full(
                    (len(xyz), 1), new_tissue.obsm["aligned_spatial"].mean(0)[0]
                )
                xyz = np.concatenate([new_column, xyz[:, :2]], axis=1)

                # filter to new tissue y,z bounds
                xyz = xyz[
                    (xyz[:, 1] > new_tissue.obsm["aligned_spatial"].min(0)[1])
                    & (xyz[:, 1] < new_tissue.obsm["aligned_spatial"].max(0)[1])
                    & (xyz[:, 2] > new_tissue.obsm["aligned_spatial"].min(0)[2])
                    & (xyz[:, 2] < new_tissue.obsm["aligned_spatial"].max(0)[2])
                ]

            else:
                xyz = self.location_model[0].sample_with_guidance(
                    adata.n_obs,
                    self.location_model[1],
                    guidance_scale=self.config["guidance_signal"],
                    use_ema=False,
                    cond_vec=np.array(
                        [
                            [
                                4,
                                5,
                                new_tissue.obsm["aligned_spatial"].mean(0)[-1],
                                0.0,
                                0.0,
                                1.0,
                            ]
                        ]
                    ),
                    small_t_threshold=15,
                )
                new_column = np.full(
                    (len(xyz), 1), new_tissue.obsm["aligned_spatial"].mean(0)[-1]
                )
                xyz = np.concatenate([xyz[:, :2], new_column], axis=1)
                xyz = xyz[
                    (xyz[:, 1] > new_tissue.obsm["aligned_spatial"].min(0)[1])
                    & (xyz[:, 1] < new_tissue.obsm["aligned_spatial"].max(0)[1])
                    & (xyz[:, 0] > new_tissue.obsm["aligned_spatial"].min(0)[0])
                    & (xyz[:, 0] < new_tissue.obsm["aligned_spatial"].max(0)[0])
                ]

            mask = np.all(np.isfinite(xyz), axis=1)
            xyz = xyz[mask]

            # Ground truth
            xyz_gt = new_tissue.obsm["aligned_spatial"].copy()

            closest_ref_slice = np.argmin(
                [
                    np.square(
                        ref_slice.obsm["aligned_spatial"].mean(0)[-1]
                        - new_tissue.obsm["aligned_spatial"].mean(0)[-1]
                    )
                    for ref_slice in self.slice_data_loader.reference_slices
                ]
            )
            cs = (
                self.slice_data_loader.reference_slices[closest_ref_slice]
                .obsm["aligned_spatial"]
                .copy()
            )
            cs[:, -1] = new_tissue.obsm["aligned_spatial"].mean(0)[-1]

        elif loc_type == "closest_slice":
            closest_ref_slice = np.argmin(
                [
                    np.square(
                        ref_slice.obsm["aligned_spatial"].mean(0)[-1]
                        - new_tissue.obsm["aligned_spatial"].mean(0)[-1]
                    )
                    for ref_slice in self.slice_data_loader.reference_slices
                ]
            )
            xyz = (
                self.slice_data_loader.reference_slices[closest_ref_slice]
                .obsm["aligned_spatial"]
                .copy()
            )
            ax = section_axis(self.config["data_mode"])
            xyz[:, ax] = new_tissue.obsm["aligned_spatial"].mean(0)[ax]


        n = adata.n_obs
        if xyz.shape[0] > n:
            idx = np.random.choice(xyz.shape[0], size=n, replace=False)
            xyz = xyz[idx]
        elif xyz.shape[0] < n:
            idx = np.random.choice(xyz.shape[0], size=n, replace=True)
            xyz = xyz[idx]

        adata.obsm["spatial"] = np.asarray(xyz)

        return adata

    # === CLUSTER / SUBCLASS ===
    def _celltype_log_prior(self, n_classes, device):
        """Log class-frequency prior over cell types for long-tail logit adjustment.

        Derived from the reference slices' tokens -- legitimately available at test
        time (same source the majority_baseline branch uses), so no GT leakage.
        Laplace-smoothed so classes unseen in the references still get a finite
        (floored) log-prior instead of -inf. Cached across calls."""
        cached = getattr(self, "_ct_log_prior_cache", None)
        if cached is not None and cached.shape[0] == n_classes:
            return cached.to(device)
        ref = ad.concat(self.slice_data_loader.reference_slices)
        tokens = np.asarray(ref.obs["token"]).astype(int)
        counts = np.bincount(tokens, minlength=n_classes).astype(np.float64)
        counts += 1.0  # Laplace smoothing: no log(0) for unseen classes
        freq = counts / counts.sum()
        log_prior = torch.tensor(np.log(freq), dtype=torch.float32, device=device)
        self._ct_log_prior_cache = log_prior
        return log_prior

    def infer_cluster(self, adata, new_tissue):
        clust_type = self.config["cluster_inference_type"]

        xyz_samples = np.asarray(adata.obsm["spatial"])

        if "majority_baseline" in clust_type:
            ref_np = np.asarray(
                ad.concat(self.slice_data_loader.reference_slices).obsm[
                    "aligned_spatial"
                ]
            )
            labels_np = np.asarray(
                ad.concat(self.slice_data_loader.reference_slices).obs["token"]
            ).astype(int)

            nbrs = NearestNeighbors(n_neighbors=20).fit(ref_np)
            _, indices = nbrs.kneighbors(xyz_samples)

            preds = np.zeros(len(xyz_samples), dtype=int)
            for i, neigh_idxs in enumerate(indices):
                counts = np.bincount(labels_np[neigh_idxs])
                preds[i] = np.argmax(counts)

            adata.obs["token"] = preds

        elif "model" in clust_type:
            xyz_samples_t = torch.tensor(
                xyz_samples, dtype=torch.float32, device=self.subclass_model.device
            )
            xyz = xyz_samples_t  # .detach().cpu().numpy()
            region_model = self.subclass_model
            region_model.eval()

            sample_from_probs = True
            # (#1) temperature scaling + (#2) long-tail logit adjustment, applied to
            # the cell-type logits before sampling. Defaults (T=1, alpha=0) reproduce
            # the previous plain-sampling behavior byte-for-byte.
            temperature = float(self.config.get("cluster_temperature", 1.0))
            prior_alpha = float(self.config.get("cluster_prior_alpha", 0.0))

            with torch.no_grad():
                xyz_tensor = torch.tensor(xyz, dtype=torch.float32).to(
                    region_model.device
                )
                input_tensor = xyz_tensor
                batch_size = 1000
                outputs = []
                for i in range(0, input_tensor.size(0), batch_size):
                    logits_batch = region_model.model(input_tensor[i : i + batch_size])
                    # logits_batch = region_model.model(input_tensor[i : i + batch_size], xyz_tensor, torch.tensor(self.slice_data_loader.test_slices[0].obsm["knn_idx"][i : i + batch_size]).cuda())
                    outputs.append(logits_batch)
                logits = torch.cat(outputs, dim=0)

                # (#1) flatten overconfident softmaxes so sampling reaches the tail.
                if temperature != 1.0:
                    logits = logits / temperature
                # (#2) subtract alpha*log(prior) to boost rare types toward their
                # reference-slice marginal abundance.
                if prior_alpha != 0.0:
                    log_prior = self._celltype_log_prior(logits.shape[1], logits.device)
                    logits = logits - prior_alpha * log_prior

                probs = torch.softmax(logits, dim=1).cpu().numpy()

                preds = np.array(
                    [
                        (
                            np.random.choice(len(p), p=p)
                            if sample_from_probs
                            else np.argmax(p)
                        )
                        for p in probs
                    ]
                )

            if isinstance(xyz, torch.Tensor):
                xyz = xyz.detach().cpu().numpy()

            adata_sampled = ad.AnnData(X=np.zeros((xyz.shape[0], 1)))
            adata_sampled.obsm["spatial"] = xyz[:, :3]
            adata_sampled.obs["token"] = preds

            adata.obs["token"] = adata_sampled.obs["token"].to_numpy()

        return adata

    def infer_expression(self, pred_data, real_data):

        exp_type = self.config["expression_inference_type"]

        if exp_type == "averaging":
            rows = np.array(
                self.token_mapping_model.get_gene_exp_from_token(
                    pred_data.obs["token"].tolist()
                )
            )[:, 0]
            obs = pred_data.obs.copy()

            # 2) make a var DataFrame of length 550
            var = pd.DataFrame(index=real_data.var_names)

            # 4) re-create
            return AnnData(X=rows, obs=obs, var=var, obsm=pred_data.obsm.copy()), None

        elif exp_type == "lookup":
            # 1) build and subsample ref_tissue as before…
            ref_tissue = ad.concat(self.slice_data_loader.reference_slices)
            if getattr(self, "ref_sample", None):
                pct = self.ref_sample
                n = int(len(ref_tissue) * pct / 100)
                ref_tissue = ref_tissue[
                    np.random.choice(len(ref_tissue), size=n, replace=False)
                ]

            # 2) extract arrays
            ref_ct = np.array(ref_tissue.obs["token"].tolist())
            ref_pos = np.array(ref_tissue.obsm["aligned_spatial"])
            try:
                ref_exp = np.array(ref_tissue.X.todense())
            except:
                ref_exp = ref_tissue.X

            pred_ct = np.array(pred_data.obs["token"].tolist())
            pred_pos = np.array(pred_data.obsm["spatial"])
            n_pred = len(pred_pos)
            n_genes = ref_exp.shape[1]

            # 3) index reference cells by type

            ref_by_type = defaultdict(list)
            for i, ct in enumerate(ref_ct):
                ref_by_type[ct].append(i)

            # 4) build one KDTree per type
            trees = {}
            for ct, idxs in ref_by_type.items():
                trees[ct] = (cKDTree(ref_pos[idxs]), idxs)

            # 5) batch‑query per type
            pred_lookup = np.zeros((n_pred, n_genes), dtype=ref_exp.dtype)
            for ct, (tree, idxs) in trees.items():
                # find all pred cells of this type
                mask = pred_ct == ct
                if not mask.any():
                    continue
                pts = pred_pos[mask]
                k = min(1, len(idxs))
                dists, nbrs = tree.query(pts, k=k)
                # ensure shape (N, k)
                if k == 1:
                    nbrs = nbrs[:, None]
                # gather and average
                selected = ref_exp[idxs][nbrs]  # shape (N, k, n_genes)
                means = selected.mean(axis=1)  # shape (N, n_genes)
                pred_lookup[mask] = means

            # 6) wrap up
            return (
                AnnData(
                    X=pred_lookup,
                    obs=pred_data.obs.copy(),
                    var=pd.DataFrame(
                        index=ref_tissue.var_names
                    ),  # ← use the original var with correct length
                    obsm=pred_data.obsm.copy(),
                ),
                None,
            )

        elif exp_type == "model":

            meta_info_path = os.path.join(self.slice_data_loader.metadata_dir, self.config["meta_info"])
            meta_info = torch.load(meta_info_path)

            # 1) preserve obs but add necessary technology metadata, etc.
            for name, default in zip(
                ["organ", "technology", "species", "disease_state"],
                ["Brain", "M550", "mouse", "healthy"],
            ):
                if name in real_data.obs.columns:
                    pred_data.obs[name] = real_data.obs[name]
                else:
                    pred_data.obs[name] = default
            # Get higher hierarchy levels

            ##IMP TO USE THIS ONE HERE
            pred_data.obs["readable_label"] = (
                self.slice_data_loader.gene_exp_model.get_label_from_token(
                    pred_data.obs["token"].values
                )
            )
            hierarchy = pkl.load(
                open(f"{self.slice_data_loader.metadata_dir}hierarchy.pkl", "rb")
            )
            for h in ["class", "subclass", "supertype", "cluster"]:
                pred_data.obs[h] = "na"
            for cell in pred_data.obs_names:
                try:
                    c, sc, st, cl = hierarchy[
                        ("cluster", pred_data.obs.loc[cell, "readable_label"])
                    ]
                except:
                    c, sc, st, cl = (
                        "01 IT-ET Glut",
                        "006 L4/5 IT CTX Glut",
                        "0027 L4/5 IT CTX Glut_5",
                        "0097 L4/5 IT CTX Glut_5",
                    )
                    print("skip", pred_data.obs.loc[cell, "readable_label"])
                for h, v in zip(
                    ["class", "subclass", "supertype", "cluster"], [c, sc, st, cl]
                ):
                    pred_data.obs.loc[cell, h] = v

            ## USE PRED DATA HERE
            pred_data.obs["x"] = pred_data.obsm["spatial"][:, 2]
            pred_data.obs["y"] = pred_data.obsm["spatial"][:, 1]
            pred_data.obs["z"] = pred_data.obsm["spatial"][:, 0]

            coordfiles = [
                f"{self.slice_data_loader.metadata_dir}edges_x.pkl",
                f"{self.slice_data_loader.metadata_dir}edges_y.pkl",
                f"{self.slice_data_loader.metadata_dir}edges_z.pkl",
            ]
            for coord, coordfile in zip(("x", "y", "z"), coordfiles):
                vals_full = pred_data.obs[f"{coord}"]
                edges = pkl.load(open(coordfile, "rb"))
                bin_idxs = np.digitize(vals_full, edges, right=True)
                pred_data.obs[f"<{coord}>"] = bin_idxs

            if self.config.get("omit_x", False) and "<x>" in pred_data.obs.columns:
                del pred_data.obs["<x>"]

            obs = pred_data.obs.copy()

            if self.config["full_gene_panel"]:
                print(f'using full gene set of {len(meta_info["gene_set"])} genes')
                var = pd.DataFrame(index=meta_info["gene_set"])
            else:
                var = pd.DataFrame(index=real_data.var_names)

            # Teacher-forcing knobs (default False => standard free-running inference).
            # cheat_with_tokens: feed GT gene identities, model only predicts values.
            # cheat_with_expr:   feed GT expression values too (fully teacher-forced).
            # expression_teacher_fill: just place GT expression in adata_sub.X so the
            #   above (or return_gt) have ground truth to read from.
            cheat_with_tokens = bool(self.config.get("cheat_with_tokens", False))
            cheat_with_expr = bool(self.config.get("cheat_with_expr", False))
            teacher_fill = bool(
                self.config.get("expression_teacher_fill", False)
                or cheat_with_tokens
                or cheat_with_expr
            )

            # 3) build X: zeros for free-running inference, or GT expression aligned
            #    onto the full gene panel when teacher forcing.
            if teacher_fill:
                real_X = real_data.X
                real_X = real_X.toarray() if hasattr(real_X, "toarray") else np.asarray(real_X)
                real_df = pd.DataFrame(real_X, columns=list(real_data.var_names))
                X_sparse = real_df.reindex(
                    columns=list(var.index), fill_value=0.0
                ).to_numpy(dtype=float)
            else:
                X_sparse = np.zeros((len(pred_data.obs_names), len(var.index)))

            # 4) re-create
            adata_sub = AnnData(
                X=X_sparse, obs=obs, var=var, obsm=pred_data.obsm.copy()
            )
            ckp_path = self.config[
                "expression_model_checkpoint"
            ]
            scml = model_inference(
                ckp_path=ckp_path,
                adata=adata_sub,
                meta_info=meta_info,
                use_kv_cache=True,
            )
            # model_inference does not set bin_edges; teacher-forced GT binning needs them.
            if teacher_fill and not hasattr(scml, "bin_edges"):
                bin_edges = compute_global_bin_edges(
                    adata_sub, meta_info["gene_set"], scml.n_express_level
                )
                scml.bin_edges = torch.tensor(bin_edges)
            results = scml.generate_cell_genesis(
                idx=range(len(pred_data.obs_names)),
                max_new_tokens=500,
                top_k=self.config.get("top_k", 5),
                verbose=False,
                return_gt=False,
                batch_size=512,
                cheat_with_tokens=cheat_with_tokens,
                cheat_with_expr=cheat_with_expr,
                fast=True,
                select_max_index_margin=self.config.get("expression_select_max_index_margin", 0.01),
            )

            if self.config.get("expression_verbose_eval", False):
                print("\n================ DEBUG: Inference Cell 0 ================")
                cell_obs = adata_sub.obs.iloc[0]
                for col in ["subclass", "supertype", "cluster", "<x>", "<y>", "<z>", "technology", "disease_state"]:
                    if col in cell_obs.index:
                        print(f"  {col}: {cell_obs[col]}")

                r0, gene_tokens_0, new_vals_0, gen_seq_0, _, _ = results[0]

                print(f"\nGenerated {len(gene_tokens_0)} tokens (first 20):")
                print(gene_tokens_0[:20])
                print("Expression vals (first 20):")
                print(new_vals_0[:20])

                gt_vec = real_data.X[0]
                if hasattr(gt_vec, "toarray"):
                    gt_vec = gt_vec.toarray().ravel()
                else:
                    gt_vec = np.asarray(gt_vec).ravel()
                gt_genes = list(real_data.var_names)
                gt_expressed = {g: v for g, v in zip(gt_genes, gt_vec) if v > 0}

                print(f"\nGT expressed genes: {len(gt_expressed)}  |  Pred expressed genes: {len(set(gene_tokens_0))}")
                print("\nTop GT expressed genes vs pred:")
                for gene, gt_val in sorted(gt_expressed.items(), key=lambda x: -x[1])[:30]:
                    flag = "✅" if gene in gene_tokens_0 else "❌"
                    pred_val = new_vals_0[gene_tokens_0.index(gene)] if gene in gene_tokens_0 else 0.0
                    print(f"  {flag} {gene:20s} | GT={gt_val:.3f} | Pred={pred_val:.3f}")
                print("===========================================================\n")

            rows = [r[0] for r in results]

            rows = np.array(rows)
            obs = pred_data.obs.copy()

            # 4) re-create
            adata = AnnData(X=rows, obs=obs, var=scml.adata.var, obsm=pred_data.obsm)
            return adata, results

    # === ORCHESTRATOR ===
    def run_inference(self, adatas):
        real_data = ad.concat(adatas)

        # start with empty shell
        pred_data = AnnData(obs=pd.DataFrame(index=real_data.obs.index))

        # step 1: location
        if "end" in self.config["location_inference_type"]:
            return pred_data
        elif "skip" in self.config["location_inference_type"]:
            pred_data.obsm["spatial"] = real_data.obsm["aligned_spatial"].copy()
        else:
            pred_data = self.infer_location(pred_data, real_data)

        plt.figure()

        plt.scatter(
            pred_data.obsm["spatial"][
                :, (2 if is_sagittal_mode(self.config["data_mode"]) else 0)
            ],
            pred_data.obsm["spatial"][:, 1],
            s=0.1,
            alpha=0.5,
        )
        plt.xlim(0, 12)
        plt.ylim(0, 8)
        plt.title("Inferred Locations model")
        plt.savefig(
            f"{self.config['artifact_dir']}/inferred_locations_model.png", dpi=300
        )

        plt.figure()

        # Ground truth
        plt.scatter(
            real_data.obsm["aligned_spatial"][
                :, (2 if is_sagittal_mode(self.config["data_mode"]) else 0)
            ],
            real_data.obsm["aligned_spatial"][:, 1],
            s=0.1,
            alpha=0.5,
        )
        plt.xlim(0, 12)
        plt.ylim(0, 8)
        plt.title("Ground Truth Locations")
        plt.savefig(
            f"{self.config['artifact_dir']}/ground_truth_locations.png", dpi=300
        )

        plt.figure()

        # step 2: cluster
        if "end" in self.config["cluster_inference_type"]:
            return pred_data
        elif "skip" in self.config["cluster_inference_type"]:
            pred_data.obs["token"] = real_data.obs["token"].copy()
        else:
            pred_data = self.infer_cluster(pred_data, real_data)

        # step 3: expression
        real_data.obsm["spatial"] = real_data.obsm["aligned_spatial"]
        assign_shared_colors([real_data, pred_data], color_key="token")
        plot_spatial_with_palette(
            real_data,
            color_key="token",
            spot_size=0.003,
            figsize=(10, 10),
            save=f"./{self.config['artifact_dir']}/real_data_clusters.png",
            saggital=is_sagittal_mode(self.config["data_mode"]),
        )
        plot_spatial_with_palette(
            pred_data,
            color_key="token",
            spot_size=0.003,
            figsize=(10, 10),
            save=f"./{self.config['artifact_dir']}/pred_data_clusters.png",
            saggital=is_sagittal_mode(self.config["data_mode"]),
        )

        if "end" in self.config["expression_inference_type"]:
            return pred_data
        elif "skip" in self.config["expression_inference_type"]:
            pred_data.X = real_data.X.copy()
        else:
            pred_data, res = self.infer_expression(pred_data, real_data)


        return pred_data
