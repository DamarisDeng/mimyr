"""Retrain the cell-type module on the Zhuang-1 dataset (data_mode base_zhuang1).

Motivation: the expression module was trained on Zhuang-1 via SliceDataLoader
(data_mode base_zhuang1), but the cell-type model in use (best_model_rq1.pt) was
trained by CombinedModel.fit() on a *different* data path (a directory of h5ad
slices, the old rq1 data). The skip ablation showed the expression module is
strong (0.62 soft_spearman given GT loc+type) while the full pipeline sits at
~0.44 -- i.e. location+cell-type generation is the bottleneck. This script
retrains ONLY the cell-type model on the same Zhuang-1 slices the expression
model saw, so its token vocabulary and tissue statistics match the rest of the
pipeline.

Uses the SAME loader cfg as the rq3_v2 distance eval (eval_spatialz_rq3distance_new.py)
so the token vocabulary is identical to what inference/eval uses. Trains
SkeletonCelltypeModel2 (aligned_spatial -> token) and saves an explicitly-named
checkpoint (the model's own fit() only writes ./best_model.pt).
"""
import argparse
import copy
import os

import numpy as np
from scipy.spatial import cKDTree

from data_loader import SliceDataLoader
from models.celltype_model import SkeletonCelltypeModel2


def compute_neighbor_tokens(adata, k, cache_path=None):
    """(N, k) int array of each cell's k nearest-neighbor token-ids, for the
    neighborhood-KL objective. k-NN is in 3D aligned_spatial: because x_ccf (the
    slice axis) is a coordinate and slices are well separated along it, 3D k-NN
    stays within-slice automatically (matches the 2D within-slice eval)."""
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape == (adata.n_obs, k):
            print(f"  loaded neighbor tokens from {cache_path} {arr.shape}")
            return arr
        print(f"  cache {cache_path} shape {arr.shape} != {(adata.n_obs, k)}; recomputing")
    coords = np.asarray(adata.obsm["aligned_spatial"], dtype=np.float32)
    tokens = adata.obs["token"].to_numpy().astype(np.int32)
    print(f"  computing {k}-NN neighbor tokens for {coords.shape[0]:,} cells (3D)...")
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1, workers=-1)  # col 0 is self (dist 0)
    nbr_tokens = tokens[idx[:, 1:]]  # drop self -> (N, k)
    if cache_path:
        np.save(cache_path, nbr_tokens)
        print(f"  cached neighbor tokens to {cache_path}")
    return nbr_tokens

# Match eval_spatialz_rq3distance_new.py exactly so the token vocab lines up with
# the rest of the pipeline at inference time.
DATA_DIR = "/work/magroup/skrieger/tissue_generator/quantized_slices"
META_DIR = "/work/magroup/skrieger/tissue_generator/spencer_mimyr/models/generative_transformer/metadata"
META_INFO = f"{META_DIR}/4hierarchy_metainfo_mouse_geneunion2_DAG.pt"
ZHUANG_DIR = "/work/magroup/skrieger/MERFISH_BICCN/processed_data"

N_CLASSES = 5274  # kept identical to the current cell-type model (SkeletonCelltypeModel2(5274))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="model_checkpoints/celltype_zhuang1.pt")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stopping patience on the (now de-noised) val soft-accuracy")
    p.add_argument("--neighborhood_k", type=int, default=20,
                   help="k for the neighborhood-KL soft label (0 disables, matches soft_accuracy k=20)")
    p.add_argument("--neighborhood_weight", type=float, default=0.0,
                   help="lambda for neighborhood-KL loss: lambda*neighborhood_CE + (1-lambda)*hard_CE. "
                        "0 = original hard-label training; 1 = pure neighborhood distribution.")
    p.add_argument("--neighbor_cache", default="",
                   help="optional .npy path to cache/reuse the (N,k) neighbor-token array")
    args = p.parse_args()

    cfg = {
        "data_dir": DATA_DIR,
        "meta_info": META_INFO,
        "data_mode": "base_zhuang1",
        "metrics": ["soft_spearman_correlation"],
        "full_gene_panel": True,
        "zhuang_data_dir": ZHUANG_DIR,
    }

    print("Loading Zhuang-1 (data_mode=base_zhuang1) via SliceDataLoader ...")
    loader = SliceDataLoader(
        mode="base_zhuang1", label="cluster", cfg=copy.deepcopy(cfg),
        metadata_dir=META_DIR, omit_x=False,
    )
    loader.prepare()

    # prepare() concatenates the per-slice lists into adata_train / adata_val and
    # then frees train_slices / val_slices (=None). No adata2 in cfg, so adata_train
    # is exactly the Zhuang-1 ST cells (coords + token), no scRNA reference mixed in.
    train_adata = loader.adata_train
    val_adata = loader.adata_val

    print(f"  train cells: {train_adata.n_obs:,}")
    print(f"  val   cells: {val_adata.n_obs:,}")

    # --- sanity checks before training ---
    coord_dim = train_adata.obsm["aligned_spatial"].shape[1]
    assert coord_dim == 3, (
        f"cell-type Model expects 3-D aligned_spatial (num_features=3), got {coord_dim}"
    )
    max_tok = int(train_adata.obs["token"].max())
    n_unique = train_adata.obs["token"].nunique()
    print(f"  token range: [0, {max_tok}]  unique tokens: {n_unique}")
    assert max_tok < N_CLASSES, (
        f"token id {max_tok} >= n_classes {N_CLASSES}; the Zhuang-1 vocab does not "
        f"fit the hardcoded class count. Re-derive n_classes before training."
    )

    neighbor_tokens = None
    if args.neighborhood_weight > 0.0 and args.neighborhood_k > 0:
        print(f"Neighborhood-KL objective: k={args.neighborhood_k} lambda={args.neighborhood_weight}")
        neighbor_tokens = compute_neighbor_tokens(
            train_adata, args.neighborhood_k,
            cache_path=(args.neighbor_cache or None),
        )

    model = SkeletonCelltypeModel2(N_CLASSES, learning_rate=args.lr)
    print(f"Training cell-type model: epochs={args.epochs} batch_size={args.batch_size} "
          f"lr={args.lr} patience={args.patience}")
    model.fit(train_adata, val_adata=val_adata, batch_size=args.batch_size,
              epochs=args.epochs, early_stop_patience=args.patience,
              neighbor_tokens=neighbor_tokens, neighborhood_weight=args.neighborhood_weight)

    # fit() only writes ./best_model.pt (and only if a val set improved); save an
    # explicitly-named checkpoint we can point --cluster_model_checkpoint at.
    model.save_model(args.out)
    print(f"Saved retrained cell-type model to {args.out}")


if __name__ == "__main__":
    main()
