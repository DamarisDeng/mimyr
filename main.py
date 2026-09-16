import argparse
from dataclasses import dataclass
from datetime import datetime
import zipfile

import gdown
import yaml

from models.combined_model import CombinedModel
from data_loader import SliceDataLoader
from data_modes import is_sagittal_mode
from models.biological_model import KDEModelForGuidance
import torch
import numpy as np
from inference import Inference
from evaluator import Evaluator
import copy, os, pandas as pd
import pickle as pkl

import json
import gc
from concurrent.futures import ThreadPoolExecutor


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@dataclass
class TrainConfig:
    degree: int = 7
    hidden_sizes: tuple = (1024, 2048, 4096, 2048, 1024)
    activation: str = "silu"
    batchnorm: bool = False
    dropout: float = 0.0
    feature_type: str = "poly"
    num_rff_features: int = 256
    rff_gamma: float = 100.0
    rff_seed: int | None = None
    n_timesteps: int = 70
    schedule_type: str = "cosine"
    beta_start: float = 1e-10
    beta_end: float = 1e-9
    cosine_s: float = 0.008
    batch_size: int = 4096 * 50
    lr: float = 2e-4
    weight_decay: float = 0
    epochs: int = 1000000
    grad_clip: float = None
    ema_decay: float = 0.999

# ----------------- argparse -----------------
def get_args():
    parser = argparse.ArgumentParser(description="Run inference with config options.")

    # config file (loaded first so its values become defaults; explicit CLI flags override them)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. Values here become defaults; CLI flags take precedence.",
    )

    # data / model paths
    parser.add_argument(
        "--run_mode", type=str, default="inference", help="Mode for SliceDataLoader", choices=["train", "inference"]
    )
    parser.add_argument(
        "--training_slice_directory", type=str, default="data/cleaned_versions", help="Training adata, if mode is set to train"
    )
    parser.add_argument(
        "--val_slice_directory", type=str, default="data/cleaned_versions", help="Validation adata, if mode is set to train"
    )
    parser.add_argument(
        "--skip_combined_fit", action="store_true",
        help="Skip combined_model.fit() (location/celltype training) and go straight to expression model training",
    )


    parser.add_argument(
        "--data_mode", type=str, default="rq1", help="Mode for SliceDataLoader"
    )
    parser.add_argument(
        "--omit_x", action="store_true", help="Omit x_ccf coordinate from aligned_spatial"
    )
    parser.add_argument(
        "--data_label",
        type=str,
        default="cluster",
        help="Label type for SliceDataLoader",
    )
    parser.add_argument(
        "--location_model_checkpoint",
        type=str,
        default="model_checkpoints/smoothtune_conditional_ddpm_2d_checkpoint_400.pt",
        help="Path to trained location checkpoint",
    )
    parser.add_argument(
        "--cluster_model_checkpoint",
        type=str,  
        default="model_checkpoints/best_model_rq1.pt",
        help="Path to trained CelltypeModel checkpoint",
    )
    parser.add_argument(
        "--expression_model_checkpoint",
        type=str,  ### CHANGE
        default="model_checkpoints/TG-base4_epoch4_model.pt",
        help="Path to trained expression checkpoint",
    )

    parser.add_argument(
        "--location_inference_type",
        type=str,
        default="skip",
        help="Type of location inference",
    )
    parser.add_argument(
        "--kde_bandwidth", type=float, default=0.01, help="Bandwidth for KDE"
    )

    parser.add_argument(
        "--cluster_inference_type",
        type=str,
        default="model",
        help="How to infer subclass",
    )
    parser.add_argument(
        "--expression_inference_type",
        type=str,
        default="end",
        help="How to infer gene expression",
    )

    # training hyperparams
    parser.add_argument(
        "--epochs", type=int, default=500, help="Training epochs for CelltypeModel"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for CelltypeModel",
    )
    parser.add_argument(
        "--guidance_signal",
        type=float,
        default=0.0,
        help="Backward-guidance strength (eta) for location diffusion: scales the "
             "gradient of the neighbouring-slice KDE that is added at each reverse "
             "step. 0 = no guidance (plain plane-conditioned sampling). Default is "
             "0 so that runs which do not ask for guidance keep the unguided "
             "behaviour every result before 2026-07 was produced with; note rows "
             "recorded before this default changed show 0.01 but were also unguided, "
             "because the guided sampler had no caller.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for CelltypeModel"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--metrics",
        type=lambda s: s.split(","),
        help="Comma-separated list of metrics to compute (soft_accuracy,soft_correlation,neighborhood_enrichment,soft_precision)",
        default=[
            "soft_accuracy"
        ],  # ,"soft_accuracy", "soft_correlation", "neighborhood_enrichment", "soft_precision"],
    )
    parser.add_argument(
        "--metric_sampling",
        type=int,
        default=1,
        help="Percentage of samples to use for metric computation",
    )
    parser.add_argument(
        "--metric_filter_by_gt",
        action="store_true",
        help="When filtering genes for expression metrics, keep genes expressed in gt only (ignoring pred)",
    )
    parser.add_argument(
        "--metric_gene_set_file",
        type=str,
        default=None,
        help="Path to a plain-text file with one gene name per line to use as the evaluation gene set (overrides the gene set from meta_info)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/output.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=1,
        help="Number of parallel CPU threads for evaluation across slices (default 1 = serial)",
    )
    parser.add_argument(
        "--meta_info",
        type=str,
        default="4hierarchy_metainfo_mouse_geneunion2_DAG.pt",
        help="meta_info file path for GE prediction",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the rq1 data (quantized slices)",
    )
    parser.add_argument(
        "--zhuang_data_dir",
        type=str,
        default=None,
        help="Base directory for Zhuang MERFISH data; must contain Zhuang-ABCA-2 and Zhuang-ABCA-3 subdirs (required for rq2, rq3, rq4 modes)",
    )
    parser.add_argument(
        "--diseased_data_dir",
        type=str,
        default=None,
        help="Directory containing diseased (5xFAD Trem2) slices (required for rq5 mode)",
    )
    parser.add_argument(
        "--rq3_v2_rq1_test_indices",
        type=str,
        default=None,
        help="Comma-separated Zhuang-ABCA-2 test slice indices for rq3_v2_rq1 mode (default: '1,10,20,30,40')",
    )

    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts",
    )

    # Expression model training args (used when --run_mode train)
    parser.add_argument(
        "--expression_output_dir",
        type=str,
        default="model_checkpoints/expression_finetuned",
        help="Directory to save the finetuned expression model",
    )
    parser.add_argument(
        "--expression_epochs", type=int, default=5,
        help="Number of fine-tuning epochs for the expression model",
    )
    parser.add_argument(
        "--expression_batch_size", type=int, default=8,
        help="Batch size for expression model training",
    )
    parser.add_argument(
        "--expression_lr", type=float, default=5e-5,
        help="Learning rate for expression model training",
    )
    parser.add_argument(
        "--expression_lambda_val", type=float, default=1.0,
        help="Weight on the expression MSE loss term",
    )
    parser.add_argument(
        "--expression_max_len", type=int, default=512,
        help="Max sequence length for prompts + genes",
    )
    parser.add_argument(
        "--expression_save_frequency", type=int, default=10,
        help="Number of epochs between expression model checkpoints",
    )
    parser.add_argument(
        "--expression_epoch_samples", type=int, default=-1,
        help="Rows to draw per epoch for expression training (-1 uses full dataset)",
    )
    parser.add_argument(
        "--expression_log_per_steps", type=int, default=100,
        help="Log to W&B every this many steps during expression training",
    )
    parser.add_argument(
        "--expression_new_expression_size", type=int, default=None,
        help="Override n_expression_level in the expression model",
    )
    parser.add_argument(
        "--expression_no_shuffle", action="store_true",
        help="Disable DataLoader shuffling for expression training",
    )
    parser.add_argument(
        "--expression_xyz_noise", action="store_true",
        help="Add noise to x,y,z coordinates during expression model training",
    )
    parser.add_argument(
        "--expression_xyz_noise_magnitude", type=int, default=2,
        help="Half-width m of the uniform discrete coordinate noise drawn from {-m..+m} "
             "when --expression_xyz_noise is set. 2 (default) reproduces the historical "
             "hard-coded {-2..2}; 0 makes the flag a no-op.",
    )
    parser.add_argument(
        "--expression_dropout", type=float, default=None,
        help="Override checkpoint dropout value (e.g. 0.0 to disable dropout during finetuning)",
    )
    parser.add_argument(
        "--expression_adata2", type=str, default=None,
        help="Optional path to a second .h5ad file (e.g. scRNA-seq) to merge into training",
    )
    parser.add_argument(
        "--expression_metadata_dir", type=str,
        default="model_checkpoints/metadata",
        help="Directory containing metadata files used by the expression model (edges_x/y/z.pkl, hierarchy.pkl, meta_info .pt)",
    )
    parser.add_argument(
        "--expression_from_finetuned", action="store_true",
        help="Indicate that the expression checkpoint is already finetuned (affects weight loading)",
    )
    parser.add_argument(
        "--expression_model_size", type=str, default=None,
        choices=["small", "medium", "large"],
        help="Model size to initialise from scratch if no expression checkpoint is provided",
    )
    parser.add_argument(
        "--expression_rebalance_only", action="store_true",
        help="Ignore CSV weights; use uniform per-cell weights then rebalance mass across st/scrna groups",
    )
    parser.add_argument(
        "--expression_eval_test", action="store_true",
        help="Also evaluate on the test split each epoch alongside validation",
    )
    parser.add_argument(
        "--expression_verbose_eval", action="store_true",
        help="Print detailed per-gene debug output for one batch per epoch during training evaluation",
    )
    parser.add_argument(
        "--expression_hidden_regressor", action="store_true",
        help="Feed transformer hidden state into epx_regressor instead of bin logits",
    )
    parser.add_argument(
        "--expression_remap_x_to_test", action="store_true",
        help="Remap training <x> bins to the nearest <x> value seen in test slices before training",
    )
    parser.add_argument(
        "--expression_continuous_coords", action="store_true",
        help="Use a linear projection for <x>/<y>/<z> tokens instead of the discrete wee embedding",
    )
    parser.add_argument(
        "--expression_save_per_cell_metrics", action="store_true",
        help="Save per-cell Pearson r and obs metadata CSV after each test evaluation epoch",
    )
    parser.add_argument(
        "--expression_preprocess_only", action="store_true",
        help="Cache train.h5ad/val.h5ad to output_dir and exit without training. No-op if cache exists.",
    )
    parser.add_argument(
        "--expression_select_max_index_margin", type=float, default=0.01,
        help="If >0, use the highest-index top-k token only when its probability is within this margin of the top probability; otherwise sample normally. 0 disables the behaviour.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Top-k sampling for expression model generation",
    )
    # --- weak teacher forcing from the spatial-lookup cell (inference-time only) ---
    parser.add_argument(
        "--expression_weak_teacher_forcing", action="store_true",
        help="Softly steer free-running expression generation toward the nearest "
             "same-cell-type reference (lookup) cell via logit biasing. Master "
             "switch; OFF is byte-identical to standard inference.",
    )
    parser.add_argument(
        "--wtf_alpha_gene", type=float, default=0.0,
        help="Logit bias added to gene-token logits the lookup prior expresses "
             "(weak teacher forcing). 0 = no gene-presence steering.",
    )
    parser.add_argument(
        "--wtf_alpha_bin", type=float, default=0.0,
        help="Logit bias added to the lookup prior's expression bin (weak teacher "
             "forcing). 0 = no bin steering.",
    )
    parser.add_argument(
        "--wtf_bin_sigma", type=float, default=0.0,
        help="If >0, the bin bias is an ordinal-aware Gaussian over bin indices "
             "(std = sigma) centered on the prior bin; 0 = single-bin spike.",
    )
    parser.add_argument(
        "--wtf_unexpressed_mode", type=str, default="none", choices=["none", "low"],
        help="How weak teacher forcing treats genes the prior does not express: "
             "'none' = no bias; 'low' = bias toward bin 0.",
    )
    parser.add_argument(
        "--ref_slab", type=float, default=None,
        help="If set, restrict the reference pool to cells within this distance (CCF "
             "mm) of the test slice along the section axis before building the "
             "nearest-neighbour KDTrees. Intended for whole-brain pools (rq4_noref) "
             "where the full pool is ~2.6M cells; trades KDTree build/query time "
             "against coverage of rare cell types. Default None = use the full pool.",
    )
    # --- tunable reference guidance: 0 = ignore reference .. 100 = exact override ---
    parser.add_argument(
        "--wtf_strength_gene", type=float, default=0.0,
        help="Reference-guidance strength for gene presence in [0,100]. Mixes the "
             "next-gene distribution toward the reference's genes in probability "
             "space; 0 = ignore, 100 = emit exactly the reference gene set.",
    )
    parser.add_argument(
        "--wtf_strength_bin", type=float, default=0.0,
        help="Reference-guidance strength for the expression bin in [0,100]. "
             "0 = ignore, 100 = use the reference's bin for each chosen gene.",
    )
    parser.add_argument(
        "--wtf_strength_value", type=float, default=0.0,
        help="Reference-guidance strength for the real expression magnitude in "
             "[0,100]. Blends the regressor value toward the reference's "
             "(normalized + log1p) expression; 0 = ignore, 100 = use it directly. "
             "This is the knob that moves mean/variance of log expression and "
             "library size.",
    )
    # --- cell-type sampling: fight mode collapse / restore local heterogeneity ---
    parser.add_argument(
        "--cluster_temperature", type=float, default=1.0,
        help="Softmax temperature applied to the cell-type logits before sampling "
             "(cluster_inference_type=model). >1 flattens overconfident distributions "
             "so sampling surfaces non-modal types; 1.0 = unchanged.",
    )
    parser.add_argument(
        "--cluster_prior_alpha", type=float, default=0.0,
        help="Long-tail logit-adjustment strength for cell-type sampling: subtract "
             "alpha*log(prior) from the logits, where prior is the class-frequency "
             "marginal of the reference slices. Boosts rare cell types toward their "
             "reference abundance. 0 = off (no adjustment), 1 = full prior removal.",
    )
    parser.add_argument(
        "--expression_ordinal_bin_sigma", type=float, default=0.0,
        help="If >0, the expression-bin loss uses a Gaussian soft target over bin indices (ordinal-aware) with this std; 0 = standard hard cross-entropy",
    )
    parser.add_argument(
        "--expression_lr_schedule", type=str, default="none", choices=["none", "cosine"],
        help="LR schedule for expression model training: 'cosine' = linear warmup then cosine decay; 'none' = constant LR",
    )
    parser.add_argument(
        "--expression_warmup_frac", type=float, default=0.0,
        help="Fraction of total training steps used for linear LR warmup (cosine schedule only)",
    )
    parser.add_argument(
        "--expression_min_lr_ratio", type=float, default=0.1,
        help="Final LR as a fraction of peak LR at the end of the cosine schedule",
    )
    # --- bin-channel scheduled sampling (exposure-bias fix on the expression-bin feedback) ---
    parser.add_argument(
        "--expression_bin_ss_eps_max", type=float, default=0.0,
        help="Max mixing prob for bin-channel scheduled sampling (fraction of generated-position "
             "bins fed back from the model's own predictions; gene tokens stay teacher-forced). "
             "0 = disabled (byte-identical to plain teacher forcing).",
    )
    parser.add_argument(
        "--expression_bin_ss_schedule", type=str, default="linear",
        choices=["constant", "linear", "sigmoid"],
        help="Epsilon ramp over epochs after warmup for bin-channel scheduled sampling.",
    )
    parser.add_argument(
        "--expression_bin_ss_warmup_epochs", type=int, default=0,
        help="Initial epochs with eps=0 (pure teacher forcing) before the bin-ss ramp.",
    )
    parser.add_argument(
        "--expression_bin_ss_feedback", type=str, default="argmax",
        choices=["argmax", "sample"],
        help="How the fed-back bin is decoded in bin-ss pass 1: argmax or sample.",
    )
    parser.add_argument(
        "--expression_bin_ss_temp", type=float, default=1.0,
        help="Temperature for --expression_bin_ss_feedback sample (ignored for argmax).",
    )

    # Two-pass: extract --config first, load YAML, set as new defaults, then re-parse
    # so that explicit CLI flags still take precedence over the config file.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Skip null values so they don't clobber argparse defaults
        file_cfg = {k: v for k, v in file_cfg.items() if v is not None}
        parser.set_defaults(**file_cfg)

    return parser.parse_args()


# ----------------- util -----------------
def write_row(row, path):
    if not os.path.isfile(path):
        pd.DataFrame([row]).to_csv(path, index=False)
        return

    existing = pd.read_csv(path)
    existing_cols = set(existing.columns)
    new_cols = set(row.keys())

    # Add missing columns to existing rows (blank) and rewrite
    for col in new_cols - existing_cols:
        existing[col] = None
    # Fill missing keys in the new row with None
    new_row = {col: row.get(col, None) for col in existing.columns}
    # Add any new columns not yet in existing
    for col in new_cols - existing_cols:
        new_row[col] = row[col]

    updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(path, index=False)


def already_done(cfg, path):
    if not os.path.isfile(path):
        return False
    existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
    check_cols = [k for k in cfg.keys() if k in existing_cols]
    df = pd.read_csv(path, usecols=check_cols)
    row = pd.Series({k: cfg[k] for k in check_cols})
    df, row = df.align(row, axis=1)
    return any((df == row).all(axis=1))


def _evaluate_slice(pred, gt_slice, cfg_copy, metadata_dir, gene_set_file, metric_sampling, metric_filter_by_gt):
    res = Evaluator(cfg_copy, metadata_dir=metadata_dir, gene_set_file=gene_set_file).evaluate(
        pred, gt_slice, sample=metric_sampling, filter_by_gt=metric_filter_by_gt,
    )
    return {k: float(v) for k, v in res.items()}


# ----------------- main -----------------


def main():
    args = get_args()
    cfg = args.__dict__


    if not os.path.exists(cfg["artifact_dir"]):
        os.makedirs(cfg["artifact_dir"], exist_ok=True)

    if not os.path.exists(cfg["data_dir"]):
        gdown.download(id="1iJX3z9S_biGCdpc-uQWwJyFImhv2mdU8", output="data.zip")
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    
    if not os.path.exists("model_checkpoints"):
        gdown.download(id="1OSh5JfXg2OXVfTIyGR33PkvQYGNqFXLD", output="model_checkpoints.zip")
        with zipfile.ZipFile("model_checkpoints.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

    print(cfg)
    slice_data_loader = SliceDataLoader(
        mode=args.data_mode,
        label=args.data_label,
        cfg=copy.deepcopy(cfg),
        metadata_dir=args.expression_metadata_dir,
        omit_x=args.omit_x,
    )

    cfg["full_gene_panel"] = True

    if args.run_mode != "train":
        slice_data_loader.prepare()
        temp_test_slices = slice_data_loader.test_slices.copy()
        temp_ref_slices = slice_data_loader.reference_slices.copy()

    artifact_dir = cfg['artifact_dir']

    combined_model = CombinedModel(
        location_model_checkpoint=args.location_model_checkpoint,
        celltype_model_checkpoint=args.cluster_model_checkpoint,
        gene_exp_model_checkpoint=args.expression_model_checkpoint,
    )

    if args.run_mode == "train":
        import torch.distributed as _dist
        os.environ["NCCL_P2P_DISABLE"] = "1"
        if _dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
            _dist.init_process_group(backend="nccl")
            _rank = _dist.get_rank()
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        else:
            _rank = 0

        if _rank == 0 and not args.skip_combined_fit:
            combined_model.fit(args.training_slice_directory, args.val_slice_directory, cfg)

        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()

        # Train the gene expression module via finetune_mimyr
        from models.generative_transformer import train_expression_model as _train_expression_model
        import argparse as _argparse

        expr_args = _argparse.Namespace(
            # paths / identity
            # An EMPTY --expression_model_checkpoint means "train from scratch": the
            # flag's argparse default is a real path, so without this coercion
            # --expression_model_size could never take effect (ckp_path was never None).
            ckp_path=args.expression_model_checkpoint or None,
            meta_info=os.path.join(args.data_dir, args.meta_info),
            output_dir=args.expression_output_dir,
            # data
            data_mode=args.data_mode,
            data_label=args.data_label,
            data_dir=args.data_dir,
            zhuang_data_dir=args.zhuang_data_dir,
            use_rq1_train=getattr(args, "use_rq1_train", False),
            adata2=args.expression_adata2,
            metadata_dir=args.expression_metadata_dir,
            # training hyperparams
            epochs=args.expression_epochs,
            batch_size=args.expression_batch_size,
            lr=args.expression_lr,
            device=args.device,
            lambda_val=args.expression_lambda_val,
            max_len=args.expression_max_len,
            no_shuffle=args.expression_no_shuffle,
            num_workers=4,
            save_frequency=args.expression_save_frequency,
            xyz_noise=args.expression_xyz_noise,
            xyz_noise_magnitude=getattr(args, "expression_xyz_noise_magnitude", 2),
            epoch_samples=args.expression_epoch_samples,
            seed=42,
            log_per_steps=args.expression_log_per_steps,
            bin_edges_file=None,
            # model init
            from_finetuned=args.expression_from_finetuned,
            model_size=args.expression_model_size,
            overwrite_vocab_size=None,
            new_expression_size=args.expression_new_expression_size,
            # sampling
            disable_sampling_probs=True,
            use_sampling_probs=False,
            sampling_col="sampling_prob",
            # misc
            dummy=False,
            kv_cache=False,
            val_split=0.0,  # data is pre-split by SliceDataLoader; this is informational only
            rebalance_only=args.expression_rebalance_only,
            eval_test=args.expression_eval_test,
            dropout=args.expression_dropout,
            verbose_eval=args.expression_verbose_eval,
            hidden_regressor=args.expression_hidden_regressor,
            remap_x_to_test=getattr(args, "expression_remap_x_to_test", False),
            continuous_coords=getattr(args, "expression_continuous_coords", False),
            omit_x=getattr(args, "omit_x", False),
            save_per_cell_metrics=getattr(args, "expression_save_per_cell_metrics", False),
            preprocess_only=getattr(args, "expression_preprocess_only", False),
            select_max_index=getattr(args, "expression_select_max_index", False),
            ordinal_bin_sigma=getattr(args, "expression_ordinal_bin_sigma", 0.0),
            lr_schedule=getattr(args, "expression_lr_schedule", "none"),
            warmup_frac=getattr(args, "expression_warmup_frac", 0.0),
            min_lr_ratio=getattr(args, "expression_min_lr_ratio", 0.1),
            # bin-channel scheduled sampling
            bin_ss_eps_max=getattr(args, "expression_bin_ss_eps_max", 0.0),
            bin_ss_schedule=getattr(args, "expression_bin_ss_schedule", "linear"),
            bin_ss_warmup_epochs=getattr(args, "expression_bin_ss_warmup_epochs", 0),
            bin_ss_feedback=getattr(args, "expression_bin_ss_feedback", "argmax"),
            bin_ss_temp=getattr(args, "expression_bin_ss_temp", 1.0),
        )
        _train_expression_model(expr_args)
        exit(0)


    # Phase 1: serial GPU inference — collect all predictions before evaluating
    inference_results = []  # list of (pred, gt_slice, cfg_copy)

    for i, slice in enumerate(temp_test_slices):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["artifact_dir"] = f"{artifact_dir}/{timestamp}"
        os.makedirs(cfg_copy["artifact_dir"], exist_ok=True)

        slice_data_loader.test_slices = [slice]

        fixed_pool = getattr(slice_data_loader, "fixed_reference_pool", False)
        if fixed_pool:
            # One whole-brain reference pool shared by every test slice (rq4_noref):
            # hand the full list through instead of the usual per-slice bracket.
            slice_data_loader.reference_slices = temp_ref_slices
        else:
            slice_data_loader.reference_slices = temp_ref_slices[
                2 * i : 2 * i + 2
            ]
            if len(slice_data_loader.reference_slices) == 0:
                slice_data_loader.reference_slices = temp_ref_slices[-2:]

        cfg_copy["slice_index"] = i

        if fixed_pool:
            # "Closest reference slice" is meaningless for a whole-brain pool drawn
            # from a differently-sectioned brain, so there is no honest KDE to build
            # here. The object is still constructed because Inference takes one, but
            # at guidance_signal == 0 sample_with_guidance short-circuits to sample()
            # and never queries it. Refuse rather than silently guide on nonsense.
            if args.guidance_signal != 0.0:
                raise ValueError(
                    f"--guidance_signal must be 0 with data_mode={args.data_mode} "
                    "(fixed reference pool): there is no single neighbouring slice to "
                    "build the backward-guidance KDE from."
                )
            pool_ref = slice_data_loader.reference_slices[0]
            n_sub = min(50_000, pool_ref.n_obs)
            best_ref_slice = pool_ref[
                np.random.choice(pool_ref.n_obs, size=n_sub, replace=False)
            ].copy()
        else:
            closest_ref_slice = np.argsort(
                [
                    np.square(
                        ref_slice.obsm["aligned_spatial"].mean(0)[-1]
                        - slice_data_loader.test_slices[0]
                        .obsm["aligned_spatial"]
                        .mean(0)[-1]
                    )
                    for ref_slice in slice_data_loader.reference_slices
                ]
            )[1]
            best_ref_slice = slice_data_loader.reference_slices[
                closest_ref_slice
            ].copy()

        if is_sagittal_mode(args.data_mode):
            best_ref_slice.obsm["aligned_spatial"][:, 0] = (
                slice_data_loader.test_slices[0].obsm["aligned_spatial"][:, 0].mean(0)
            )
        else:
            best_ref_slice.obsm["aligned_spatial"] = best_ref_slice.obsm[
                "aligned_spatial"
            ][:, :2]

        kdemodel = KDEModelForGuidance(
            [best_ref_slice], bandwidth=args.kde_bandwidth
        )
        kdemodel.fit()

        if already_done(cfg_copy, args.out_csv):
            print("skip", cfg_copy)
            continue

        inf = Inference(
            combined_model,
            kdemodel,
            slice_data_loader,
            cfg_copy,
        )

        pred = inf.run_inference(slice_data_loader.test_slices)
        print("Inference done for slice", i, pred)
        inference_results.append((pred, slice, cfg_copy))
        del inf
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 2: parallel CPU evaluation
    with ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_slice,
                pred, gt_slice, cfg_copy,
                args.expression_metadata_dir,
                args.metric_gene_set_file,
                args.metric_sampling,
                args.metric_filter_by_gt,
            )
            for pred, gt_slice, cfg_copy in inference_results
        ]
        eval_results = [f.result() for f in futures]

    # Phase 3: serial write and artifact save
    for (pred, gt_slice, cfg_copy), res in zip(inference_results, eval_results):
        row = {**cfg_copy, **res}
        write_row(row, args.out_csv)
        print("wrote", cfg_copy)

        cfg_path = os.path.join(cfg_copy["artifact_dir"], "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg_copy, f, indent=2)

        res_path = os.path.join(cfg_copy["artifact_dir"], "results.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

        pred_path = os.path.join(cfg_copy["artifact_dir"], "pred.pkl")
        with open(pred_path, "wb") as f:
            pkl.dump(pred, f)

    del inference_results, eval_results
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
