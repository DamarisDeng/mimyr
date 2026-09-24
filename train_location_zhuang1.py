"""Retrain the location DDPM on the Zhuang-1 dataset (data_mode base_zhuang1).

Motivation: the bisection (results/rq3_distance_ablation_skipmodelmodel.csv) showed
the location module is the dominant bottleneck -- it costs ~0.10 soft_spearman
(skip/model/model ~0.56 vs full model/model/model ~0.46), roughly 2x the cell-type
module's cost. Given GT locations, Mimyr already BEATS SpatialZ at every distance,
so fixing locations is what closes the d1/d2 gap. The DDPM currently in use
(smoothtune_conditional_ddpm_2d_checkpoint_400.pt) was trained by an external
script on the old rq1 data; this retrains it on the same Zhuang-1 slices the
expression + cell-type modules saw.

The DDPMTrainer.train() loop already exists in models/diffusion_model.py; it was
just never driven with real data (CombinedModel only ever builds it with
(None, None, cfg) to load a checkpoint and sample()). This script supplies the
data and calls train().

AXIS CONVENTION (critical): aligned_spatial is stacked as
[z_ccf, y_ccf, x_ccf]  (data_loader.py:_align_and_tokenize_slices), so column -1
is x_ccf, the anterior-posterior / slice-separating axis. DDPMTrainer trains on
coords[:, :2] = [z_ccf, y_ccf] (the in-slice plane) and treats column -1 (x_ccf)
as the depth conditioning. Inference conditions location on
[4, 5, aligned_spatial.mean(0)[-1], 0, 0, 1] = [4, 5, x_ccf, 0, 0, 1]
(inference.py infer_location, the rq3_v2_d* fall-through branch). We therefore
feed aligned_spatial AS-IS and build cond = [4, 5, x_ccf, 0, 0, 1] per cell so the
training conditioning matches how the model is sampled.

NORMALIZATION: DDPMTrainer hardcodes coord/cond mean+std, and inference
un-normalizes with those SAME hardcoded constants (CombinedModel loads only
model+ema from the checkpoint, never the saved stats). Normalization only needs
to be *consistent* between train and inference, not "correct", so we leave the
overrides untouched -- the retrained model will be sampled with the same constants
it was trained with.
"""
import argparse
import copy

import numpy as np

from data_loader import SliceDataLoader
from models.diffusion_model import DDPMTrainer
# IMPORTANT: use the SAME TrainConfig that CombinedModel builds the model with at
# inference (models/combined_model.py), NOT models.diffusion_model.TrainConfig.
# They differ in ways that MUST match between train and inference:
#   - arch: degree=7, hidden_sizes=(1024,2048,4096,2048,1024), feature_type="poly",
#           activation="silu"  (the diffusion_model default is a broken ((512,512),))
#   - schedule: n_timesteps=70 (the diffusion_model default is 1000 -> sampling with
#           70 steps would be garbage)
# The production checkpoint loads exactly into this arch (verified via load_state_dict).
from models.combined_model import TrainConfig

DATA_DIR = "/work/pi_f008n64_dartmouth_edu/Mimyr/data"
META_DIR = "/work/pi_f008n64_dartmouth_edu/Mimyr/spencer_code/models/generative_transformer/metadata"
META_INFO = f"{META_DIR}/4hierarchy_metainfo_mouse_geneunion2_DAG.pt"
ZHUANG_DIR = "/work/pi_f008n64_dartmouth_edu/Mimyr/data"


def build_cond(aligned_spatial):
    """cond = [4, 5, x_ccf, 0, 0, 1] per cell, matching the rq3_v2_d* inference
    conditioning. x_ccf is aligned_spatial[:, -1] (the slice-depth axis)."""
    n = aligned_spatial.shape[0]
    x_ccf = np.asarray(aligned_spatial)[:, -1]
    cond = np.zeros((n, 6), dtype=np.float32)
    cond[:, 0] = 4.0
    cond[:, 1] = 5.0
    cond[:, 2] = x_ccf
    cond[:, 3] = 0.0
    cond[:, 4] = 0.0
    cond[:, 5] = 1.0
    return cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ckpt_prefix", default="location_zhuang1_ddpm_2d_checkpoint",
                   help="checkpoint basename; epoch is appended, saved under model_checkpoints/")
    args = p.parse_args()

    cfg_data = {
        "data_dir": DATA_DIR,
        "meta_info": META_INFO,
        "data_mode": "base_zhuang1",
        "metrics": ["soft_spearman_correlation"],
        "full_gene_panel": True,
        "zhuang_data_dir": ZHUANG_DIR,
    }

    print("Loading Zhuang-1 (data_mode=base_zhuang1) via SliceDataLoader ...")
    loader = SliceDataLoader(
        mode="base_zhuang1", label="cluster", cfg=copy.deepcopy(cfg_data),
        metadata_dir=META_DIR, omit_x=False,
    )
    loader.prepare()
    train_adata = loader.adata_train

    coords_np = np.asarray(train_adata.obsm["aligned_spatial"], dtype=np.float32)  # [z_ccf, y_ccf, x_ccf]
    assert coords_np.shape[1] == 3, f"expected 3-D aligned_spatial, got {coords_np.shape}"
    cond_np = build_cond(coords_np)

    print(f"  train cells: {coords_np.shape[0]:,}")
    # sanity: the in-plane axes (trained) and the depth axis (conditioned)
    print(f"  in-plane z_ccf: mean={coords_np[:,0].mean():.3f} std={coords_np[:,0].std():.3f}")
    print(f"  in-plane y_ccf: mean={coords_np[:,1].mean():.3f} std={coords_np[:,1].std():.3f}")
    print(f"  depth   x_ccf: mean={coords_np[:,2].mean():.3f} std={coords_np[:,2].std():.3f} "
          f"(range {coords_np[:,2].min():.3f}..{coords_np[:,2].max():.3f})")
    print("  NOTE: DDPMTrainer normalizes coords with hardcoded mean~[5.07,5.12] std~[2.26,2.94]; "
          "large deviations here just mean larger normalized targets (still consistent w/ inference).")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_prefix=args.ckpt_prefix,
        viz=False,  # headless: skip the in-loop 200k-point sample figures
    )
    print(f"Training location DDPM: epochs={cfg.epochs} batch_size={cfg.batch_size} "
          f"lr={cfg.lr} ckpt_prefix={cfg.ckpt_prefix} (saves every 20 epochs)")

    trainer = DDPMTrainer(coords_np, cond_np, cfg)
    trainer.train()
    print(f"Done. Checkpoints at model_checkpoints/{cfg.ckpt_prefix}_<epoch>.pt")


if __name__ == "__main__":
    main()
