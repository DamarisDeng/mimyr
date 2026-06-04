#!/usr/bin/env python3
"""
finetune.py

Fine-tune MimyrModel on conditional generation with coordinate tokens,
including a held-out validation split.
"""
import os
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import scanpy as sc
from tqdm import tqdm
import wandb
import torch.distributed as dist
import scipy.sparse as sp
import pandas as pd
from anndata import AnnData
from concurrent.futures import ThreadPoolExecutor


from .model.model import MimyrConfig, MimyrModel
from .utils.hf_tokenizer import MimyrTokenizer
from .data_util import get_generation_dataloader, harmonize_dataset, summarize_sample_ddp
from .Mimyr import tokens_and_vals_to_expression_row

from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict


def load_sampling_metadata_csv(
    adata: AnnData, path: str, overwrite: bool = True
) -> None:
    """
    Load the CSV saved above and merge into adata.obs by obs_names.
    """
    df = pd.read_csv(path)
    WEIGHT_COLS = ["class_weight", "spatial_bin", "cell_weight", "sampling_prob"]
    if "obs_name" not in df.columns:
        raise ValueError("CSV must contain an 'obs_name' column.")
    df = df.set_index("obs_name")

    # keep only rows present in adata
    df = df.loc[df.index.intersection(adata.obs_names)]
    if df.empty:
        print("No overlapping obs_names; nothing to merge.")
        return

    for c in df.columns:
        if c not in WEIGHT_COLS:
            continue
        if overwrite or c not in adata.obs.columns:
            adata.obs.loc[df.index, c] = df[c].values
        else:
            # fill NaNs only
            m = adata.obs.index.isin(df.index) & adata.obs[c].isna()
            adata.obs.loc[m, c] = df.loc[adata.obs.index[m], c].values
    print(f"Merged {list(df.columns)} for {df.shape[0]} cells from {path}")


def rebalance_sampling_mass(
    adata_train,
    group_col="dataset_type",
    sampling_col="sampling_prob",
    target_fracs=None,
    fill_missing_with=1.0,
    eps=1e-12,
):
    """
    Rescales sampling_prob so that total probability mass per group matches target_fracs.
    Keeps relative weights within each group unchanged.
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

    scales = {}
    for g, t in target.items():
        m = max(cur.get(g, 0.0), eps)
        scales[g] = t / m

    gvals = adata_train.obs[group_col].to_numpy()
    scale_vec = np.vectorize(lambda g: scales.get(g, 1.0))(gvals)
    adata_train.obs[sampling_col] = adata_train.obs[sampling_col].to_numpy() * scale_vec
    print("rebalanced sampling probs")
    return scales


def evaluate_expression_correlation(
    adata,
    batch,
    logits_labels,
    logits_exp_real,
    tokenizer,
    tokens_and_vals_to_expression_row_fn,
    var_names,
    labels,
    mask_token_id=-100,
    threshold=0.0,
):
    """
    Compute per-cell Pearson correlation between predicted and true expression.

    Parameters
    ----------
    adata : AnnData
        The original data object to get ground truth from.
    batch : dict
        The current batch from the dataloader.
    logits_labels : Tensor
        Output of the model (B, T, vocab_size) — token logits.
    logits_exp_real : Tensor
        Output of the model (B, T, 1) — real-valued expression predictions.
    tokenizer : MimyrTokenizer
        Tokenizer used to map tokens to strings.
    tokens_and_vals_to_expression_row_fn : function
        The function that builds a gene expression vector from tokens and values.
    var_names : List[str]
        List of all gene names in order.
    mask_token_id : int
        The ID of the mask/pad token to ignore in output.

    Returns
    -------
    pearson_rs : List[float]
        Pearson correlation per cell in the batch.
    """
    B, T, V = logits_labels.shape
    logits_cls = logits_labels.argmax(dim=-1).cpu().numpy()  # (B, T)
    expr_vals = logits_exp_real.squeeze(-1).cpu().numpy()  # (B, T)
    if hasattr(labels, "detach"):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)
    idxs = batch["idx"] if "idx" in batch else range(B)  # indices into adata

    predicted_expr_rows = []
    ground_truth_rows = []

    for b in range(B):
        gene_token_ids = logits_cls[b]
        expr_values = expr_vals[b]
        labels_b = labels_np[b]  # (T,)

        # Filter out special tokens (mask, EOS, etc.)
        valid_mask = labels_b != mask_token_id
        gene_token_ids = gene_token_ids[valid_mask]
        expr_values = expr_values[valid_mask]

        # Convert token IDs to strings
        gene_tokens = tokenizer.convert_ids_to_tokens(gene_token_ids.tolist())
        # if b == 0:
        #     print('labels')
        #     print(labels_b)
        #     print(valid_mask)
        #     print('gene token ids')
        #     print(gene_token_ids)
        #     print('gene tokens')
        #     print(gene_tokens)
        #     print('expression values:')
        #     print(expr_values)

        # Construct predicted expression row
        pred_expr = tokens_and_vals_to_expression_row_fn(
            var_names=var_names,
            gene_tokens=gene_tokens,
            gene_tokens_int=gene_token_ids.tolist(),
            new_vals=expr_values.tolist(),
            return_series=False,
        )
        # if b == 0:
        #     print('pred_expr')
        #     print(pred_expr)
        #     for v, p in zip(var_names,pred_expr):
        #         if p > 0:
        #             print(f'{v}: {p}')
        predicted_expr_rows.append(pred_expr)

        # Get true expression from AnnData (dense or sparse)
        gt_expr = (
            adata.X[idxs[b]].toarray().flatten()
            if hasattr(adata.X, "toarray")
            else adata.X[idxs[b]]
        )
        ground_truth_rows.append(gt_expr)

    # Compute per-cell Pearson r
    pearson_rs = []
    for pred, gt in zip(predicted_expr_rows, ground_truth_rows):
        if np.std(pred) > 0 and np.std(gt) > 0:
            r, _ = pearsonr(pred, gt)
        else:
            r = 0.0
        pearson_rs.append(r)

    return pearson_rs


def evaluate_expression_metrics(
    adata,
    batch,
    logits_labels,  # (B, T, V) torch.Tensor
    logits_exp_real,  # (B, T, 1) torch.Tensor
    tokenizer,
    tokens_and_vals_to_expression_row_fn,
    var_names,
    labels,  # (B, T) torch.Tensor or np.ndarray; ignore positions == mask_token_id
    mask_token_id=-100,
    threshold=0.0,  # expression > threshold => positive
    verbose=False,
):
    """
    Returns per-cell metrics: Pearson r, F1, precision, recall.
    """
    import numpy as np
    from scipy.stats import pearsonr

    gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()

    # Align sums to provided var_names (in case orders differ)
    # Fast path if identical:
    if list(getattr(adata, "var_names", [])) == list(var_names):
        valid_gene_mask = gene_sums > 0
    else:
        # Map adata.var_names -> index
        name_to_idx = {g: i for i, g in enumerate(list(adata.var_names))}
        valid_gene_mask = np.array(
            [
                (gene_sums[name_to_idx[g]] > 0) if g in name_to_idx else False
                for g in var_names
            ],
            dtype=bool,
        )

    B, T, V = logits_labels.shape
    logits_cls = logits_labels.argmax(dim=-1).detach().cpu().numpy()  # (B, T)
    expr_vals = logits_exp_real.squeeze(-1).detach().cpu().numpy()  # (B, T)

    if hasattr(labels, "detach"):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    idxs = batch.get("idx", range(B))

    pearson_rs, f1s, precisions, recalls = [], [], [], []
    for b in range(B):
        ids = logits_cls[b]
        vals = expr_vals[b]
        lab = labels_np[b]

        # Keep only positions we trained on (mask out prompt/pad/etc.)
        keep = lab != mask_token_id
        if not np.any(keep):
            # No usable tokens -> zero-vector prediction
            pred_expr = np.zeros(len(var_names), dtype=float)
        else:
            ids = ids[keep]
            vals = vals[keep]
            tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
            pred_expr = tokens_and_vals_to_expression_row_fn(
                var_names=var_names,
                gene_tokens=tokens,
                gene_tokens_int=ids.tolist(),
                new_vals=vals.tolist(),
                return_series=False,
                max_violations=10,
            )

        # Ground truth
        gt = adata.X[idxs[b]]
        gt = gt.toarray().ravel() if hasattr(gt, "toarray") else np.asarray(gt).ravel()

        pred_expr = pred_expr[valid_gene_mask]
        gt = gt[valid_gene_mask]
        valid_genes = np.array(var_names)[valid_gene_mask]

        # Pearson r
        if np.std(pred_expr) > 0 and np.std(gt) > 0:
            r, _ = pearsonr(pred_expr, gt)
        else:
            r = 0.0
        pearson_rs.append(float(r))

        # F1 / Precision / Recall (binary by threshold)
        pred_pos = pred_expr > threshold
        gt_pos = gt > threshold

        tp = int(np.count_nonzero(pred_pos & gt_pos))
        pp = int(np.count_nonzero(pred_pos))
        gp = int(np.count_nonzero(gt_pos))

        prec = tp / pp if pp > 0 else 0.0
        rec = tp / gp if gp > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))
        if b == 0 and verbose == True:
            print("\n================ DEBUG: Cell 0 ================")
            if "input_ids" in batch:
                input_ids_b0 = batch["input_ids"][0]
                if hasattr(input_ids_b0, "detach"):
                    input_ids_b0 = input_ids_b0.detach().cpu().numpy()
                else:
                    input_ids_b0 = np.asarray(input_ids_b0)
                prompt_mask = labels_np[0] == mask_token_id
                prompt_ids = input_ids_b0[prompt_mask]
                prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids.tolist())
                prompt_vals = batch["input_vals"][0][prompt_mask]
                print(f"prompt token ids ({len(prompt_ids)} tokens):")
                print(prompt_ids)
                print("prompt tokens:")
                print(prompt_tokens)
                print("prompt vals:")
                print(prompt_vals)
            print("labels")
            print(lab)
            print("gene token ids")
            print(ids)
            print("gene tokens")
            print(tokens)
            print("expression values:")
            print(vals)
            print("pred_expr")
            print(pred_expr)
            # for v, p in zip(var_names[valid_gene_mask],pred_expr):
            #     if p > 0:
            #         print(f'{v}: {p}')

            gt_expressed_idx = np.where(gt > 0)[0]
            print(
                f"Ground-truth expressed genes: {len(gt_expressed_idx)} / {len(valid_genes)}"
            )

            for i in gt_expressed_idx:
                gene = valid_genes[i]
                gt_val = gt[i]
                pred_val = pred_expr[i]

                present_in_pred = gene in tokens
                pred_idx = tokens.index(gene) if gene in tokens else -1
                pred_val_raw = vals[pred_idx] if pred_idx != -1 else 0.0
                flag = "✅" if present_in_pred else "❌"
                print(
                    f"{flag} {gene:15s} | GT={gt_val:.3f} | Pred={pred_val:.3f} | Pred2={pred_val_raw:.3f}"
                )

            # Genes predicted >0 but GT==0
            false_pos_idx = np.where((gt == 0) & (pred_expr > 0))[0]
            false_pos_genes = valid_genes[false_pos_idx]
            if len(false_pos_genes) > 0:
                print("\nPredicted >0 but GT=0 (false positives):")
                print(", ".join(false_pos_genes.tolist()))
            else:
                print("\nNo false positives detected.")
            print("================================================\n")

    return pearson_rs, f1s, precisions, recalls


# ---------------------------------------------------------------------------
# Fast vectorized evaluation helpers
# ---------------------------------------------------------------------------

def _build_token_to_var_idx(tokenizer, var_names):
    """Build (vocab_size,) int32 array mapping token_id → var_names index (-1 if not a gene)."""
    var_to_idx = {g: i for i, g in enumerate(var_names)}
    vocab_size = len(tokenizer)
    mapping = np.full(vocab_size, -1, dtype=np.int32)
    all_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    for tid, tok in enumerate(all_tokens):
        if tok is not None and tok in var_to_idx:
            mapping[tid] = var_to_idx[tok]
    return mapping


def _build_valid_gene_mask(adata, var_names):
    """Boolean mask over var_names for genes with nonzero expression in adata."""
    gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()
    if list(adata.var_names) == list(var_names):
        return gene_sums > 0
    name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    return np.array(
        [(gene_sums[name_to_idx[g]] > 0) if g in name_to_idx else False for g in var_names],
        dtype=bool,
    )


def _evaluate_metrics_vectorized(
    adata_X,
    idxs,
    logits_labels,       # (B, T, V) CPU tensor  — ignored when token_preds_np is given
    logits_exp_real,     # (B, T, 1) CPU tensor  — ignored when expr_vals_np is given
    labels,              # (B, T) numpy int array; -100 marks prompt/pad
    token_to_var_idx,    # (vocab_size,) int32; -1 = not a gene
    valid_gene_mask,     # (n_genes,) bool
    n_genes,
    mask_token_id=-100,
    threshold=0.0,
    token_preds_np=None,  # pre-computed (B, T) int64 numpy array (argmax already done on GPU)
    expr_vals_np=None,    # pre-computed (B, T) float32 numpy array (squeeze already done on GPU)
):
    """Fully-vectorized batch evaluation: Pearson r, F1, precision, recall."""
    if token_preds_np is not None:
        B = token_preds_np.shape[0]
        token_preds = token_preds_np
    else:
        B = logits_labels.shape[0]
        token_preds = logits_labels.argmax(dim=-1).numpy()         # (B, T) int64
    expr_vals = expr_vals_np if expr_vals_np is not None else logits_exp_real.squeeze(-1).float().numpy()  # (B, T) float32

    # Build predicted expression matrix via scatter — no Python per-cell loop
    valid_pos = labels != mask_token_id                  # (B, T)
    b_idxs, t_idxs = np.where(valid_pos)
    tids      = token_preds[b_idxs, t_idxs]
    vals      = expr_vals[b_idxs, t_idxs]
    gene_idxs = token_to_var_idx[tids]

    keep = gene_idxs >= 0
    b_k  = b_idxs[keep]
    g_k  = gene_idxs[keep]
    v_k  = vals[keep]

    pred_expr = np.zeros((B, n_genes), dtype=np.float32)
    count_mat = np.zeros((B, n_genes), dtype=np.int32)
    flat_idx  = b_k * n_genes + g_k
    np.add.at(pred_expr.ravel(), flat_idx, v_k)
    np.add.at(count_mat.ravel(), flat_idx, 1)
    nz = count_mat > 0
    pred_expr[nz] /= count_mat[nz]

    # Ground-truth: single batch fetch instead of per-cell access
    idxs_list = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
    if sp.issparse(adata_X):
        gt_expr = np.asarray(adata_X[idxs_list].todense(), dtype=np.float32)
    else:
        gt_expr = np.asarray(adata_X[idxs_list], dtype=np.float32)

    p = pred_expr[:, valid_gene_mask]  # (B, G)
    g = gt_expr[:,  valid_gene_mask]   # (B, G)

    # Vectorized Pearson correlation
    p_c   = p - p.mean(axis=1, keepdims=True)
    g_c   = g - g.mean(axis=1, keepdims=True)
    denom = np.sqrt((p_c ** 2).sum(axis=1)) * np.sqrt((g_c ** 2).sum(axis=1))
    pearson_rs = np.where(denom > 0, (p_c * g_c).sum(axis=1) / denom, 0.0)

    # Vectorized F1 / precision / recall
    pred_pos = p > threshold
    gt_pos   = g > threshold
    tp = (pred_pos & gt_pos).sum(axis=1).astype(np.float32)
    pp = pred_pos.sum(axis=1).astype(np.float32)
    gp = gt_pos.sum(axis=1).astype(np.float32)
    prec = np.where(pp > 0, tp / pp, 0.0)
    rec  = np.where(gp > 0, tp / gp, 0.0)
    f1   = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)

    return pearson_rs.tolist(), f1.tolist(), prec.tolist(), rec.tolist()


def _remap_x_to_test_distribution(adata_train: AnnData, adata_test: AnnData, seed: int = 42) -> None:
    """
    Slice-aware stochastic remap of training <x> bins to test <x> bins.

    Each coronal slice spans a small range of x_bins (not a single value).
    The original global nearest-neighbor remap collapsed multiple training
    x_bins to a few test x_bins, leaving most of each test slice's range OOD.

    This version:
      1. Auto-detects slice groups by finding large gaps in sorted unique x_bins.
      2. Matches training groups to test groups via linear_sum_assignment.
      3. Randomly samples new x_bins from the full target test slice range.
    """
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)

    def _detect_slice_groups(xbins, min_gap_multiplier=2.0):
        unique = np.sort(np.unique(xbins))
        if len(unique) < 2:
            return [unique]
        diffs = np.diff(unique)
        threshold = diffs.mean() * min_gap_multiplier
        boundaries = np.where(diffs >= threshold)[0] + 1
        return np.split(unique, boundaries)

    train_groups = _detect_slice_groups(adata_train.obs["<x>"].values.astype(int))
    test_groups  = _detect_slice_groups(adata_test.obs["<x>"].values.astype(int))

    print(f"  Detected {len(train_groups)} train slice x_bin groups: {[g.tolist() for g in train_groups]}")
    print(f"  Detected {len(test_groups)}  test  slice x_bin groups: {[g.tolist() for g in test_groups]}")

    train_means = np.array([g.mean() for g in train_groups])
    test_means  = np.array([g.mean() for g in test_groups])

    cost = np.abs(train_means[:, None] - test_means[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    print(f"\n  Optimal training→test assignment (total cost {cost[row_ind, col_ind].sum():.2f}):")

    new_x = adata_train.obs["<x>"].values.astype(int).copy()
    train_xbin = adata_train.obs["<x>"].values.astype(int)
    matched_test_groups = set(col_ind)

    for r, c in zip(row_ind, col_ind):
        tr_grp = train_groups[r]
        te_grp = test_groups[c]
        mask = np.isin(train_xbin, tr_grp)
        n = mask.sum()
        new_x[mask] = rng.choice(te_grp, size=n, replace=True)
        print(f"    train {tr_grp.tolist()} (mean={train_means[r]:.1f}, n={n})"
              f" → test {te_grp.tolist()} (mean={test_means[c]:.1f})")

    adata_train.obs["<x>"] = new_x

    unmatched = [i for i in range(len(test_groups)) if i not in matched_test_groups]
    if unmatched:
        print(f"\n  Warning: {len(unmatched)} test slice group(s) have no matched training slice:")
        for i in unmatched:
            print(f"    {test_groups[i].tolist()} (mean={test_means[i]:.1f}) — x_bins remain OOD")
    else:
        print("\n  All test slice groups are covered.")

    train_x_after = np.unique(new_x)
    test_x_all = np.unique(adata_test.obs["<x>"].values.astype(int))
    uncovered = np.setdiff1d(test_x_all, train_x_after)
    print(f"\n  Train x_bins after remap ({len(train_x_after)}): {train_x_after.tolist()}")
    print(f"  Uncovered test x_bins ({len(uncovered)}): {uncovered.tolist()}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune MimyrModel on conditional generation with coords + validation"
    )
    parser.add_argument(
        "--ckp-path", type=str, default=None, help="Path to pretrained checkpoint (.pt)"
    )
    parser.add_argument(
        "--meta-info",
        type=str,
        required=True,
        help="Path to meta_info.pt from pretraining",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        required=True,
        help="Data mode to pass to SliceDataLoader (e.g. 'rq1', 'rq2', 'rq3', 'rq4', 'rq5')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing the .h5ad data files",
    )
    parser.add_argument(
        "--zhuang-data-dir",
        type=str,
        default=None,
        help="Base directory for Zhuang MERFISH data; must contain Zhuang-ABCA-2 and Zhuang-ABCA-3 subdirs (required for rq2, rq3, rq4 modes)",
    )
    parser.add_argument(
        "--data-label",
        type=str,
        default="cluster",
        help="Cell-type label column to use in SliceDataLoader (default: 'cluster')",
    )
    parser.add_argument(
        "--adata2",
        type=str,
        default=None,
        help="Path to second input AnnData .h5ad file",
    )
    parser.add_argument(
        "--kv-cache", action="store_true", help="Whether to use kv-cached model variant"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save finetuned model and tokenizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of fine-tuning epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--ordinal-bin-sigma", dest="ordinal_bin_sigma", type=float, default=0.0,
        help="If >0, expression-bin loss uses a Gaussian soft target over bin indices (ordinal-aware) with this std; 0 = hard cross-entropy",
    )
    # --- bin-channel scheduled sampling (exposure-bias fix on the expression-bin feedback) ---
    parser.add_argument(
        "--bin-ss-eps-max", dest="bin_ss_eps_max", type=float, default=0.0,
        help="Max mixing prob for bin-channel scheduled sampling: fraction of generated-position "
             "bins replaced by the model's own predicted bin (gene tokens stay teacher-forced). "
             "0 (default) disables the feature -> training is byte-identical to plain teacher forcing.",
    )
    parser.add_argument(
        "--bin-ss-schedule", dest="bin_ss_schedule", type=str, default="linear",
        choices=["constant", "linear", "sigmoid"],
        help="How epsilon ramps over epochs after warmup: constant=eps_max throughout; "
             "linear=linear ramp to eps_max by the final epoch; sigmoid=inverse-sigmoid ramp.",
    )
    parser.add_argument(
        "--bin-ss-warmup-epochs", dest="bin_ss_warmup_epochs", type=int, default=0,
        help="Number of initial epochs with eps=0 (pure teacher forcing) before the ramp starts.",
    )
    parser.add_argument(
        "--bin-ss-feedback", dest="bin_ss_feedback", type=str, default="argmax",
        choices=["argmax", "sample"],
        help="How the fed-back bin is decoded in pass 1: argmax (matches greedy inference) or "
             "sample from softmax(logits / bin-ss-temp).",
    )
    parser.add_argument(
        "--bin-ss-temp", dest="bin_ss_temp", type=float, default=1.0,
        help="Temperature for --bin-ss-feedback sample (ignored for argmax).",
    )
    parser.add_argument(
        "--lr-schedule", dest="lr_schedule", type=str, default="none", choices=["none", "cosine"],
        help="LR schedule: 'cosine' = linear warmup then cosine decay; 'none' = constant LR",
    )
    parser.add_argument(
        "--warmup-frac", dest="warmup_frac", type=float, default=0.0,
        help="Fraction of total steps for linear LR warmup (cosine only)",
    )
    parser.add_argument(
        "--min-lr-ratio", dest="min_lr_ratio", type=float, default=0.1,
        help="Final LR as a fraction of peak LR at end of cosine schedule",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Max sequence length for prompts + genes",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable DataLoader shuffling"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=10,
        help="Number of epochs between model checkpoints",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=1.0,
        help="Weight on expression MSE loss term",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cpu or cuda)",
    )
    parser.add_argument(
        "--from-finetuned",
        action="store_true",
        help="Indicate checkpoint already includes finetuned vocab size",
    )
    parser.add_argument(
        "--overwrite-vocab-size",
        type=int,
        default=None,
        help="If set, overwrite the model and config vocab_size to this value before loading state_dict",
    )
    parser.add_argument(
        "--new-expression-size",
        type=int,
        default=None,
        help="If set, overwrite the model and config n_expression_level to this value",
    )
    parser.add_argument(
        "--xyz-noise",
        action="store_true",
        help="Whether to add noise to x,y,z coordinates during training",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Whether to use small dummy dataset"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default=None,
        help="Model size to initialize if no checkpoint is provided",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="model_checkpoints/metadata",
        help="Directory containing metadata files (edges_x/y/z.pkl, meta_info .pt) for harmonization",
    )
    parser.add_argument(
        "--disable-sampling-probs",
        action="store_true",
        help="Ignore adata.obs[sampling_col] even if present (uniform sampling).",
    )
    parser.add_argument(
        "--rebalance-only",
        action="store_true",
        help="Ignore CSV weights; set uniform per-cell weights then rebalance dataset mass across groups (st vs scrna).",
    )
    parser.add_argument(
        "--sampling-col",
        type=str,
        default="sampling_prob",
        help="obs column name holding per-cell sampling probabilities",
    )
    parser.add_argument(
        "--epoch-samples",
        type=int,
        default=-1,
        help="Rows to draw per epoch (set -1 to use len(dataset))",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-per-steps",
        type=int,
        default=100,
        help="Logging to WANDB only after this many steps",
    )
    parser.add_argument(
        "--bin-edges-file",
        type=str,
        default=None,
        help="bin_edges file for gene expression binning",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Also evaluate on the test split each epoch alongside validation",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override the dropout value from the checkpoint model_args (e.g. 0.0 to disable dropout)",
    )
    parser.add_argument(
        "--verbose-eval",
        action="store_true",
        help="Print detailed per-gene debug output for one batch per epoch during training evaluation",
    )
    parser.add_argument(
        "--hidden-regressor",
        action="store_true",
        help="Feed transformer hidden state (n_embd) into epx_regressor instead of bin logits",
    )
    parser.add_argument(
        "--remap-x-to-test",
        action="store_true",
        help=(
            "Before training, remap each training cell's binned <x> value to the nearest "
            "binned <x> value present in the test slices. Requires the data mode to have "
            "a test split (slice_loader.adata_test must be non-None)."
        ),
    )
    parser.add_argument(
        "--save-per-cell-metrics",
        action="store_true",
        help=(
            "After each test evaluation epoch, save a CSV with per-cell Pearson r "
            "and obs metadata (cluster, x_ccf, y_ccf, z_ccf, x_bin, y_bin, z_bin) "
            "to <output_dir>/per_cell_metrics_epoch<N>.csv. Useful for generalization "
            "gap analysis."
        ),
    )
    parser.add_argument(
        "--omit-x",
        action="store_true",
        help="Omit the x_ccf coordinate from aligned_spatial in the anndatas",
    )
    parser.add_argument(
        "--continuous-coords",
        action="store_true",
        help=(
            "Use a linear projection for <x>/<y>/<z> coordinate tokens instead of "
            "the discrete wee embedding table, so coordinate values generalise continuously."
        ),
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help=(
            "Preprocess and cache train.h5ad / val.h5ad to output_dir, then exit without "
            "training. If the cache files already exist, exits immediately. Run this "
            "single-threaded before the multi-GPU training job to avoid OOM during data prep."
        ),
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP). By default AMP is enabled on CUDA.",
    )
    parser.add_argument(
        "--select-max-index",
        action="store_true",
        help="At each generation step, pick the highest-index token among top-k candidates instead of sampling.",
    )

    return parser


def _bin_ss_eps(epoch, args):
    """Mixing probability epsilon for bin-channel scheduled sampling at this epoch (1-indexed).

    Returns 0 (pure teacher forcing) when the feature is disabled or during warmup. After
    warmup, ramps from 0 -> bin_ss_eps_max over the remaining epochs per --bin-ss-schedule.
    """
    eps_max = getattr(args, "bin_ss_eps_max", 0.0)
    if eps_max <= 0.0:
        return 0.0
    warmup = getattr(args, "bin_ss_warmup_epochs", 0)
    if epoch <= warmup:
        return 0.0
    sched = getattr(args, "bin_ss_schedule", "linear")
    if sched == "constant":
        return eps_max
    # progress in (0, 1] over the post-warmup epochs
    span = max(1, args.epochs - warmup)
    p = min(1.0, (epoch - warmup) / span)
    if sched == "sigmoid":
        # inverse-sigmoid decay of the teacher-forcing rate (Bengio et al. 2015), normalized
        # so p=0 -> ~0 and p=1 -> eps_max.
        import math
        k = 6.0  # steepness; ~symmetric ramp centered at p=0.5
        s = lambda z: 1.0 / (1.0 + math.exp(-k * (z - 0.5)))
        lo, hi = s(0.0), s(1.0)
        return eps_max * (s(p) - lo) / (hi - lo)
    return eps_max * p  # linear


@torch.no_grad()
def _make_bin_ss_input(model, input_ids, x_expr, labels, eps, feedback, temp, use_amp):
    """Build the (partly) self-generated expression-bin input for scheduled sampling.

    Pass 1: forward with GT gene tokens + GT bins to get the model's own next-bin predictions,
    then replace a Bernoulli(eps) fraction of *generated* positions (labels != -100) with the
    model's predicted bin. Prompt and padding positions always keep the GT bin. Gene tokens are
    never touched. Returns x_expr unchanged when eps <= 0 (no extra forward pass).
    """
    if eps <= 0.0:
        return x_expr
    net = model.module if hasattr(model, "module") else model
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = net(idx=input_ids, x_expr=x_expr, return_hidden=False)
    logits_exp_bins = out[1].float()  # (B, T, num_bins)
    if feedback == "sample":
        B, T, K = logits_exp_bins.shape
        probs = F.softmax(logits_exp_bins / max(temp, 1e-6), dim=-1).reshape(-1, K)
        pred = torch.multinomial(probs, num_samples=1).reshape(B, T)
    else:  # argmax
        pred = logits_exp_bins.argmax(dim=-1)  # (B, T)
    # logits_exp_bins[:, j] predicts the bin at position j+1, so shift right to align with input.
    cand = x_expr.clone()
    cand[:, 1:] = pred[:, :-1]
    # Only corrupt generated positions; keep prompt (labels == -100) and padding as GT.
    gen_mask = labels != -100
    flip = (torch.rand_like(x_expr, dtype=torch.float) < eps) & gen_mask
    return torch.where(flip, cand, x_expr)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if dist.is_available() and dist.is_initialized():
        print("using a distributed multi-gpu run\n\n\n")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        device = torch.device(args.device)

    # Fast early exit: if --preprocess-only and cache already exists, nothing to do.
    def _find_cache(stem):
        for ext in (".slaf", ".h5ad"):
            p = os.path.join(args.output_dir, stem + ext)
            if os.path.exists(p):
                return p
        return None

    if getattr(args, "preprocess_only", False) and _find_cache("train") and _find_cache("val"):
        if rank == 0:
            print(f"Cache already exists at {args.output_dir}. Nothing to do.")
        return

    # 1) Load pretrained checkpoint
    if args.ckp_path is not None:
        ckp = torch.load(args.ckp_path, map_location="cpu")
    else:
        # Choose default config based on --model-size
        if args.model_size is None:
            raise ValueError(
                "You must specify --model-size if --ckp_path is not provided"
            )

        if args.model_size == "small":
            model_args = {
                "n_embd": 64,
                "n_layer": 2,
                "n_head": 4,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        elif args.model_size == "medium":
            model_args = {
                "n_embd": 384,
                "n_layer": 12,
                "n_head": 12,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        elif args.model_size == "large":
            model_args = {
                "n_embd": 1120,
                "n_layer": 24,
                "n_head": 16,
                "vocab_size": 1011,
                "expression_level": 100,
                "dropout": 0.1,
                "ele": 1,
            }
        else:
            raise ValueError("Invalid --model-size. Choose from: small, medium, large")

        ckp = {"model_args": model_args}
    if args.overwrite_vocab_size is not None:
        ckp["model_args"]["vocab_size"] = args.overwrite_vocab_size
        print(f"Overwriting config.vocab_size to {args.overwrite_vocab_size}")
    if args.new_expression_size is not None and args.from_finetuned:
        ckp["model_args"]["expression_level"] = args.new_expression_size
    if args.dropout is not None:
        ckp["model_args"]["dropout"] = args.dropout
    if getattr(args, "hidden_regressor", False):
        ckp["model_args"]["hidden_regressor"] = True
    if getattr(args, "continuous_coords", False):
        ckp["model_args"]["continuous_coords"] = True
    if getattr(args, "ordinal_bin_sigma", 0.0):
        ckp["model_args"]["ordinal_bin_sigma"] = args.ordinal_bin_sigma
    gptconf = MimyrConfig(**ckp["model_args"])
    print(gptconf)
    ModelClass = MimyrModel
    model = ModelClass(gptconf)
    # device = torch.device(args.device)

    if args.from_finetuned:
        model.load_state_dict(ckp["model"], strict=False)
    model.eval()
    model.hidden_dim = ckp["model_args"]["n_embd"]

    meta_info = torch.load(args.meta_info)
    print("loaded meta_info")

    # 3) Initialize tokenizer and resize model embeddings/output
    tokenizer = MimyrTokenizer(meta_info["token_set"])
    if not args.from_finetuned:
        sep = meta_info.get("sep_token", "<SPToken1>")
        tokenizer.add_special_tokens({"sep_token": sep})
        # resize_token_embeddings comes from PreTrainedModel
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        ckp["model_args"]["vocab_size"] = len(tokenizer)
        if args.new_expression_size:
            model.resize_expression_embeddings(args.new_expression_size)
            model.config.expression_level = args.new_expression_size
            ckp["model_args"]["expression_level"] = args.new_expression_size

    if ckp["model_args"].get("continuous_coords"):
        coord_token_ids = tokenizer.convert_tokens_to_ids(["<x>", "<y>", "<z>"])
        ckp["model_args"]["coord_token_ids"] = coord_token_ids
        model.config.coord_token_ids = coord_token_ids

    print("initialized model", flush=True)

    # 4) Load train/val AnnData from data_loader (or cache if available)
    import scanpy as _sc
    from data_loader import SliceDataLoader

    train_cache = _find_cache("train")
    val_cache = _find_cache("val")
    cache_exists = train_cache is not None and val_cache is not None

    if cache_exists:
        if rank == 0:
            print(f"Loading preprocessed adata from cache in {args.output_dir} ({os.path.basename(train_cache)}, {os.path.basename(val_cache)})")
        from data_loader import _read_slice
        adata_train = _read_slice(train_cache)
        adata_val = _read_slice(val_cache)
        adata_test = None
    else:
        cfg = {
            "data_dir": args.data_dir,
            "meta_info": os.path.basename(args.meta_info),
            "zhuang_data_dir": getattr(args, "zhuang_data_dir", None),
            "use_rq1_train": getattr(args, "use_rq1_train", False),
        }
        slice_loader = SliceDataLoader(
            mode=args.data_mode,
            label=args.data_label,
            cfg=cfg,
            metadata_dir=args.metadata_dir,
            omit_x=getattr(args, "omit_x", False),
        )
        slice_loader.prepare(
            adata2_path=args.adata2,
            gene_set=meta_info.get("gene_set"),
            seed=args.seed,
            dummy=args.dummy,
            sampling_col=args.sampling_col,
            output_dir=args.output_dir,
            rank=rank,
        )
        adata_train = slice_loader.adata_train
        adata_val = slice_loader.adata_val
        adata_test = slice_loader.adata_test if args.eval_test else None

        if rank == 0:
            # _build_adata writes train.h5ad when adata2 is set; write it here if not already done
            _train_write = os.path.join(args.output_dir, "train.h5ad")
            if not os.path.exists(_train_write):
                adata_train.write(_train_write)
            adata_val.write(os.path.join(args.output_dir, "val.h5ad"))
            print(f"Cached preprocessed adata to {args.output_dir}")

    if getattr(args, "preprocess_only", False):
        if rank == 0:
            print("--preprocess-only: data cached, exiting before training.")
        return

    if getattr(args, "remap_x_to_test", False):
        remap_test_adata = slice_loader.adata_test
        if remap_test_adata is None:
            raise ValueError(
                "--remap-x-to-test requires the data mode to provide a test split, "
                "but slice_loader.adata_test is None."
            )
        _remap_x_to_test_distribution(adata_train, remap_test_adata)

    args.use_sampling_probs = not args.disable_sampling_probs
    if args.rebalance_only:
        print("Mode: --rebalance-only. Using uniform per-cell weights then rebalancing dataset mass.")
        adata_train.obs[args.sampling_col] = 1.0
        args.use_sampling_probs = True
        if "dataset_type" in adata_train.obs.columns:
            rebalance_sampling_mass(
                adata_train,
                group_col="dataset_type",
                sampling_col=args.sampling_col,
                target_fracs={"st": 0.7, "scrna": 0.3},
            )
    elif args.use_sampling_probs:
        load_sampling_metadata_csv(adata_train, "train_adata_sampling.csv.gz", overwrite=True)

    use_probs_val = False  # usually evaluate uniformly

    # Optional: if you want to limit rows per epoch, pass args.epoch_samples if you have it
    epoch_samples = getattr(args, "epoch_samples", None)
    base_seed = getattr(args, "seed", 0)

    if args.bin_edges_file is not None:
        bin_edges = torch.load(args.bin_edges_file)
    else:
        bin_edges = None

    train_loader, bin_edges = get_generation_dataloader(
        adata=adata_train,
        meta_info=meta_info,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        n_express_level=model.config.expression_level,
        include_0s=False,
        add_xyz_noise=args.xyz_noise,
        # exclude_columns = ['supertype','cluster'],
        use_sampling_probs=args.use_sampling_probs,
        sampling_col=args.sampling_col,
        epoch_samples=epoch_samples,
        seed=base_seed,
        bin_edges=bin_edges,
    )
    if not args.bin_edges_file:
        torch.save(bin_edges, f"{args.output_dir}bin_edges.pt")

    if adata_val is not None:
        val_loader, _ = get_generation_dataloader(
            adata=adata_val,
            meta_info=meta_info,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
            num_workers=args.num_workers,
            n_express_level=model.config.expression_level,
            include_0s=False,
            # keep uniform sampling for validation
            use_sampling_probs=use_probs_val,
            sampling_col=args.sampling_col,
            bin_edges=bin_edges,
        )
    else:
        val_loader = None

    if adata_test is not None:
        test_loader, _ = get_generation_dataloader(
            adata=adata_test,
            meta_info=meta_info,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
            num_workers=args.num_workers,
            n_express_level=model.config.expression_level,
            include_0s=False,
            use_sampling_probs=False,
            sampling_col=args.sampling_col,
            bin_edges=bin_edges,
        )
    else:
        test_loader = None
    print("loaded anndata")

    # 5) Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Optional LR schedule: linear warmup then cosine decay to (min_lr_ratio * lr),
    # stepped once per optimizer step. "none" (default) keeps the constant LR.
    lr_schedule = getattr(args, "lr_schedule", "none")
    scheduler = None
    if lr_schedule == "cosine":
        import math as _math
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = int(getattr(args, "warmup_frac", 0.0) * total_steps)
        min_ratio = getattr(args, "min_lr_ratio", 0.1)

        def _lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + _math.cos(_math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        if rank == 0:
            print(f"LR schedule: cosine, total_steps={total_steps}, "
                  f"warmup_steps={warmup_steps}, min_lr={args.lr * min_ratio:.2e}")

    model.to(device)
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    use_amp = not getattr(args, "no_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 6) Initialize W&B
    if rank == 0:
        wandb.init(
            project="Mimyr-finetune",
            name=f"{os.path.basename(args.output_dir)}_bs{args.batch_size}",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lambda_val": args.lambda_val,
                "bin_ss_eps_max": args.bin_ss_eps_max,
                "bin_ss_schedule": args.bin_ss_schedule,
                "bin_ss_warmup_epochs": args.bin_ss_warmup_epochs,
                "bin_ss_feedback": args.bin_ss_feedback,
                "bin_ss_temp": args.bin_ss_temp,
            },
            dir=args.output_dir,
        )

    if args.from_finetuned:
        # e.g. ckp-path ends with ".../epoch3_model.pt"
        import re

        m = re.search(r"epoch(\d+)", os.path.basename(args.ckp_path))
        start_epoch = int(m.group(1)) + 1 if m else 1
    else:
        start_epoch = 1

    # Pre-compute evaluation helpers once (token→gene index map + valid-gene masks)
    var_names_train = adata_train.var_names.tolist()
    token_to_var_idx = _build_token_to_var_idx(tokenizer, var_names_train)
    valid_gene_mask_train = _build_valid_gene_mask(adata_train, var_names_train)
    n_genes_train = len(var_names_train)
    if adata_val is not None:
        valid_gene_mask_val = _build_valid_gene_mask(adata_val, adata_val.var_names.tolist())
        token_to_var_idx_val = _build_token_to_var_idx(tokenizer, adata_val.var_names.tolist())
        n_genes_val = adata_val.n_vars
    if adata_test is not None:
        valid_gene_mask_test = _build_valid_gene_mask(adata_test, adata_test.var_names.tolist())
        token_to_var_idx_test = _build_token_to_var_idx(tokenizer, adata_test.var_names.tolist())
        n_genes_test = adata_test.n_vars

    # Background thread for rank-0 metric computation so the main thread (and GPU)
    # is never blocked waiting for CPU eval work during DDP training.
    _metric_executor = ThreadPoolExecutor(max_workers=1)
    _metric_futures = []

    # 7) Training + validation loop
    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Bin-channel scheduled sampling: fraction of generated-position bins fed back from
        # the model's own predictions this epoch (0 = pure teacher forcing).
        bin_ss_eps = _bin_ss_eps(epoch, args)
        if rank == 0 and bin_ss_eps > 0.0:
            print(f"Epoch {epoch} — bin-channel scheduled sampling eps={bin_ss_eps:.3f} "
                  f"(feedback={args.bin_ss_feedback})")

        # Grab the *local* indices from the sampler, then aggregate
        local_idx = None
        if hasattr(train_loader.sampler, "get_last_indices"):
            # Need to trigger one iterator to make __iter__ run at least once
            _ = iter(train_loader)
            local_idx = train_loader.sampler.get_last_indices()
            summarize_sample_ddp(
                adata_train,
                local_idx,
                cell_key="cluster",
                group_key="dataset_type",
                top_k=20,
            )

        model.train()
        total_loss, total_cls, total_exp_bin, total_exp_real = 0.0, 0.0, 0.0, 0.0
        all_train_pearson_rs = []
        all_train_f1, all_train_prec, all_train_rec = [], [], []
        verbose_done_this_epoch = False
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            x_expr = batch["input_vals"].to(device)
            expr_target = batch["target_vals"].to(device)

            if labels is not None:
                labels = labels.to(device)

            # Pass 1 (no-grad): build the (partly) self-generated bin input. Returns x_expr
            # unchanged when bin_ss_eps == 0, so plain teacher forcing skips the extra forward.
            x_expr_in = _make_bin_ss_input(
                model, input_ids, x_expr, labels, bin_ss_eps,
                args.bin_ss_feedback, args.bin_ss_temp, use_amp,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                (
                    logits_cls,
                    logits_exp_bins,
                    logits_exp_real,
                    loss,
                    loss_cls,
                    loss_exp_bin,
                    loss_exp_real,
                ) = model(
                    idx=input_ids,
                    x_expr=x_expr_in,
                    x_expr_target=(x_expr if bin_ss_eps > 0.0 else None),
                    targets=labels,
                    y_expr=expr_target,
                    lambda_val=args.lambda_val,
                    return_hidden=False,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            # Single GPU→CPU sync for loss scalars (reused below to avoid a second sync)
            _loss_val     = loss.item()     if loss     is not None else 0.0
            _cls_val      = loss_cls.item() if loss_cls is not None else 0.0
            _exp_bin_val  = loss_exp_bin.item() if loss_exp_bin  is not None else 0.0
            _exp_real_val = loss_exp_real.item() if loss_exp_real is not None else 0.0

            total_loss     += _loss_val
            total_cls      += _cls_val
            total_exp_bin  += _exp_bin_val
            total_exp_real += _exp_real_val

            if rank == 0 and step % args.log_per_steps == 0:
                do_verbose = getattr(args, "verbose_eval", False) and not verbose_done_this_epoch
                if do_verbose:
                    # Synchronous verbose path (rare — only once per epoch)
                    with torch.no_grad():
                        pearson_rs_train, f1s, precs, recs = evaluate_expression_metrics(
                            adata=adata_train,
                            batch=batch,
                            logits_labels=logits_cls.detach().cpu(),
                            logits_exp_real=logits_exp_real.detach().cpu(),
                            tokenizer=tokenizer,
                            tokens_and_vals_to_expression_row_fn=tokens_and_vals_to_expression_row,
                            var_names=adata_train.var_names.tolist(),
                            labels=labels,
                            mask_token_id=-100,
                            verbose=True,
                        )
                    verbose_done_this_epoch = True
                    mean_r   = float(np.mean(pearson_rs_train)) if pearson_rs_train else 0.0
                    mean_f1  = float(np.mean(f1s))   if f1s   else 0.0
                    mean_prc = float(np.mean(precs)) if precs else 0.0
                    mean_rec = float(np.mean(recs))  if recs  else 0.0
                    all_train_pearson_rs.extend(pearson_rs_train)
                    all_train_f1.extend(f1s)
                    all_train_prec.extend(precs)
                    all_train_rec.extend(recs)
                    wandb.log({
                        "train/batch_loss":          _loss_val,
                        "train/batch_loss_cls":      _cls_val,
                        "train/batch_loss_exp_bin":  _exp_bin_val,
                        "train/batch_loss_exp_real": _exp_real_val,
                        "train/bin_ss_eps":          bin_ss_eps,
                        "train/pearson_r": mean_r,
                        "train/f1":        mean_f1,
                        "train/precision": mean_prc,
                        "train/recall":    mean_rec,
                        "train/step": (epoch - 1) * len(train_loader) + step,
                    })
                else:
                    # Async path: snapshot CPU tensors and hand off to background thread.
                    # Main thread (and GPU) proceed immediately — no blocking on eval.
                    _snap_logits = logits_cls.detach().cpu()
                    _snap_expr   = logits_exp_real.detach().cpu()
                    _snap_lnp    = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
                    _snap_idx    = batch.get("idx", range(logits_cls.shape[0]))
                    if hasattr(_snap_idx, "cpu"):
                        _snap_idx = _snap_idx.cpu()
                    _snap_loss  = (_loss_val, _cls_val, _exp_bin_val, _exp_real_val)
                    _snap_step  = (epoch - 1) * len(train_loader) + step

                    def _metric_task(sl, se, lnp, idx, loss_snap, log_step):
                        rs, f1s, precs, recs = _evaluate_metrics_vectorized(
                            adata_X=adata_train.X,
                            idxs=idx,
                            logits_labels=sl,
                            logits_exp_real=se,
                            labels=lnp,
                            token_to_var_idx=token_to_var_idx,
                            valid_gene_mask=valid_gene_mask_train,
                            n_genes=n_genes_train,
                        )
                        wandb.log({
                            "train/batch_loss":          loss_snap[0],
                            "train/batch_loss_cls":      loss_snap[1],
                            "train/batch_loss_exp_bin":  loss_snap[2],
                            "train/batch_loss_exp_real": loss_snap[3],
                            "train/pearson_r": float(np.mean(rs))    if rs    else 0.0,
                            "train/f1":        float(np.mean(f1s))   if f1s   else 0.0,
                            "train/precision": float(np.mean(precs)) if precs else 0.0,
                            "train/recall":    float(np.mean(recs))  if recs  else 0.0,
                            "train/step": log_step,
                        })
                        return rs, f1s, precs, recs

                    _metric_futures.append(_metric_executor.submit(
                        _metric_task,
                        _snap_logits, _snap_expr, _snap_lnp, _snap_idx, _snap_loss, _snap_step,
                    ))

        # Drain background metric futures before computing epoch-level stats.
        # By now most will already be done; this is just a final sync point.
        if rank == 0:
            for fut in _metric_futures:
                try:
                    rs, f1s, precs, recs = fut.result()
                    all_train_pearson_rs.extend(rs)
                    all_train_f1.extend(f1s)
                    all_train_prec.extend(precs)
                    all_train_rec.extend(recs)
                except Exception as e:
                    print(f"Warning: background metric task failed: {e}")
            _metric_futures.clear()

        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_exp_bin = total_exp_bin / len(train_loader)
        avg_exp_real = total_exp_real / len(train_loader)
        print(
            f"Epoch {epoch} — train total {avg_loss:.4f}, cls {avg_cls:.4f}, exp_bin {avg_exp_bin:.4f}, exp_real {avg_exp_real:.4f}"
        )
        # Accumulate epoch-level metrics; val metrics will be merged before the single log call
        epoch_log = {}
        if rank == 0:
            mean_train_pearson_epoch = (
                float(np.mean(all_train_pearson_rs))
                if len(all_train_pearson_rs)
                else 0.0
            )
            mean_train_f1_epoch = float(np.mean(all_train_f1)) if all_train_f1 else 0.0
            mean_train_prc_epoch = (
                float(np.mean(all_train_prec)) if all_train_prec else 0.0
            )
            mean_train_rec_epoch = (
                float(np.mean(all_train_rec)) if all_train_rec else 0.0
            )
            epoch_log.update(
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_loss_cls": avg_cls,
                    "train/epoch_loss_exp_bin": avg_exp_bin,
                    "train/epoch_loss_exp_real": avg_exp_real,
                    "train/pearson_r_epoch": mean_train_pearson_epoch,
                    "train/f1_epoch": mean_train_f1_epoch,
                    "train/precision_epoch": mean_train_prc_epoch,
                    "train/recall_epoch": mean_train_rec_epoch,
                    "epoch": epoch,
                }
            )

        if val_loader is not None:
            model.eval()
            v_loss, v_cls, v_exp_bin, v_exp_real, count = 0.0, 0.0, 0.0, 0.0, 0
            _val_tok_preds, _val_expr_vals, _val_labels, _val_idxs = [], [], [], []
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels")
                    x_expr = batch["input_vals"].to(device)
                    expr_target = batch["target_vals"].to(device)

                    if labels is not None:
                        labels = labels.to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits_labels, _, logits_exp_real, l, lc, leb, ler = model(
                            idx=input_ids,
                            x_expr=x_expr,
                            targets=labels,
                            y_expr=expr_target,
                            lambda_val=args.lambda_val,
                            return_hidden=False,
                        )
                    v_loss += l.item()
                    v_cls += lc.item()
                    v_exp_bin += leb.item()
                    v_exp_real += ler.item()
                    count += 1
                    # Compute argmax/squeeze on GPU before transfer — reduces (B,T,V)→(B,T),
                    # cutting GPU→CPU bandwidth ~500x vs transferring full logits.
                    _tok_np = logits_labels.argmax(dim=-1).cpu().numpy()
                    _exp_np = logits_exp_real.squeeze(-1).float().cpu().numpy()
                    _lnp_val = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
                    _idx_val = batch.get("idx", range(logits_labels.shape[0]))
                    if hasattr(_idx_val, "cpu"):
                        _idx_val = _idx_val.cpu()
                    _val_tok_preds.append(_tok_np)
                    _val_expr_vals.append(_exp_np)
                    _val_labels.append(_lnp_val)
                    _val_idxs.append(
                        _idx_val if isinstance(_idx_val, np.ndarray) else np.asarray(list(_idx_val))
                    )

            # GPU loop is done; compute metrics once over the full val set.
            # One sparse slice is far cheaper than 216 per-batch sparse slices.
            _all_tok = np.concatenate(_val_tok_preds, axis=0)
            _all_exp = np.concatenate(_val_expr_vals, axis=0)
            _all_lnp = np.concatenate(_val_labels, axis=0)
            _all_idx = np.concatenate(_val_idxs, axis=0)
            all_pearson_rs, all_val_f1, all_val_prec, all_val_rec = _evaluate_metrics_vectorized(
                adata_X=adata_val.X,
                idxs=_all_idx,
                logits_labels=None,
                logits_exp_real=None,
                labels=_all_lnp,
                token_to_var_idx=token_to_var_idx_val,
                valid_gene_mask=valid_gene_mask_val,
                n_genes=n_genes_val,
                token_preds_np=_all_tok,
                expr_vals_np=_all_exp,
            )

            avg_v_loss = v_loss / count
            avg_v_cls = v_cls / count
            avg_v_exp_bin = v_exp_bin / count
            avg_v_exp_real = v_exp_real / count
            print(
                f"Epoch {epoch} — valid total {avg_v_loss:.4f}, cls {avg_v_cls:.4f}, exp_bin {avg_v_exp_bin:.4f}, exp_real {avg_v_exp_real:.4f}"
            )
            mean_pearson_r = np.mean(all_pearson_rs)
            print(f"Validation mean Pearson r: {mean_pearson_r:.4f}")
            mean_val_f1 = float(np.mean(all_val_f1)) if all_val_f1 else 0.0
            mean_val_prc = float(np.mean(all_val_prec)) if all_val_prec else 0.0
            mean_val_rec = float(np.mean(all_val_rec)) if all_val_rec else 0.0
            print(
                f"Validation mean F1: {mean_val_f1:.4f} (P={mean_val_prc:.4f}, R={mean_val_rec:.4f})"
            )
            if rank == 0:
                epoch_log.update(
                    {
                        "valid/epoch_loss": avg_v_loss,
                        "valid/epoch_loss_cls": avg_v_cls,
                        "valid/epoch_loss_exp_bin": avg_v_exp_bin,
                        "valid/epoch_loss_exp_real": avg_v_exp_real,
                        "valid/pearson_r": mean_pearson_r,
                        "valid/f1": mean_val_f1,
                        "valid/precision": mean_val_prc,
                        "valid/recall": mean_val_rec,
                    }
                )

        if test_loader is not None:
            model.eval()
            t_loss, t_cls, t_exp_bin, t_exp_real, t_count = 0.0, 0.0, 0.0, 0.0, 0
            _test_tok_preds, _test_expr_vals, _test_labels, _test_idxs = [], [], [], []
            with torch.inference_mode():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch} [test]"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels")
                    x_expr = batch["input_vals"].to(device)
                    expr_target = batch["target_vals"].to(device)

                    if labels is not None:
                        labels = labels.to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits_labels, _, logits_exp_real, l, lc, leb, ler = model(
                            idx=input_ids,
                            x_expr=x_expr,
                            targets=labels,
                            y_expr=expr_target,
                            lambda_val=args.lambda_val,
                            return_hidden=False,
                        )
                    t_loss += l.item()
                    t_cls += lc.item()
                    t_exp_bin += leb.item()
                    t_exp_real += ler.item()
                    t_count += 1
                    _tok_np = logits_labels.argmax(dim=-1).cpu().numpy()
                    _exp_np = logits_exp_real.squeeze(-1).float().cpu().numpy()
                    _lnp_test = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
                    _idx_test = batch.get("idx", range(logits_labels.shape[0]))
                    if hasattr(_idx_test, "cpu"):
                        _idx_test = _idx_test.cpu()
                    _test_tok_preds.append(_tok_np)
                    _test_expr_vals.append(_exp_np)
                    _test_labels.append(_lnp_test)
                    _test_idxs.append(
                        _idx_test if isinstance(_idx_test, np.ndarray) else np.asarray(list(_idx_test))
                    )

            # GPU loop done; compute metrics once over the full test set
            _all_test_tok = np.concatenate(_test_tok_preds, axis=0)
            _all_test_exp = np.concatenate(_test_expr_vals, axis=0)
            _all_test_lnp = np.concatenate(_test_labels, axis=0)
            _all_test_idx = np.concatenate(_test_idxs, axis=0)
            all_test_pearson_rs, all_test_f1, all_test_prec, all_test_rec = _evaluate_metrics_vectorized(
                adata_X=adata_test.X,
                idxs=_all_test_idx,
                logits_labels=None,
                logits_exp_real=None,
                labels=_all_test_lnp,
                token_to_var_idx=token_to_var_idx_test,
                valid_gene_mask=valid_gene_mask_test,
                n_genes=n_genes_test,
                token_preds_np=_all_test_tok,
                expr_vals_np=_all_test_exp,
            )
            all_test_cell_idxs = _all_test_idx.tolist()

            avg_t_loss = t_loss / t_count
            avg_t_cls = t_cls / t_count
            avg_t_exp_bin = t_exp_bin / t_count
            avg_t_exp_real = t_exp_real / t_count
            print(
                f"Epoch {epoch} — test total {avg_t_loss:.4f}, cls {avg_t_cls:.4f}, exp_bin {avg_t_exp_bin:.4f}, exp_real {avg_t_exp_real:.4f}"
            )
            mean_test_pearson_r = np.mean(all_test_pearson_rs)
            print(f"Test mean Pearson r: {mean_test_pearson_r:.4f}")
            mean_test_f1 = float(np.mean(all_test_f1)) if all_test_f1 else 0.0
            mean_test_prc = float(np.mean(all_test_prec)) if all_test_prec else 0.0
            mean_test_rec = float(np.mean(all_test_rec)) if all_test_rec else 0.0
            print(
                f"Test mean F1: {mean_test_f1:.4f} (P={mean_test_prc:.4f}, R={mean_test_rec:.4f})"
            )
            if rank == 0:
                epoch_log.update(
                    {
                        "test/epoch_loss": avg_t_loss,
                        "test/epoch_loss_cls": avg_t_cls,
                        "test/epoch_loss_exp_bin": avg_t_exp_bin,
                        "test/epoch_loss_exp_real": avg_t_exp_real,
                        "test/pearson_r": mean_test_pearson_r,
                        "test/f1": mean_test_f1,
                        "test/precision": mean_test_prc,
                        "test/recall": mean_test_rec,
                    }
                )
                if getattr(args, "save_per_cell_metrics", False) and adata_test is not None:
                    meta_cols = [c for c in ["cluster", "x_ccf", "y_ccf", "z_ccf", "<x>", "<y>", "<z>", "slice_idx"]
                                 if c in adata_test.obs.columns]
                    pcm = pd.DataFrame({
                        "cell_idx": all_test_cell_idxs,
                        "pearson_r": all_test_pearson_rs,
                    })
                    for col in meta_cols:
                        pcm[col] = adata_test.obs[col].iloc[all_test_cell_idxs].values
                    pcm_path = os.path.join(args.output_dir, f"per_cell_metrics_epoch{epoch}.csv")
                    pcm.to_csv(pcm_path, index=False)
                    print(f"  Saved per-cell metrics → {pcm_path}")

        # Log all epoch-level metrics (train + val + test) in one call so they share the same W&B step
        if rank == 0:
            wandb.log(epoch_log)

        # Save epoch checkpoint
        if epoch % args.save_frequency == 0 and rank == 0:
            ckpt_file = os.path.join(args.output_dir, f"epoch{epoch}_model.pt")
            if dist.is_available() and dist.is_initialized():
                torch.save(
                    {
                        "model": model.module.state_dict(),
                        "model_args": ckp["model_args"],
                    },
                    ckpt_file,
                )
            else:
                torch.save(
                    {"model": model.state_dict(), "model_args": ckp["model_args"]},
                    ckpt_file,
                )

    # 8) Save final artifacts
    # model.save_pretrained(args.output_dir)
    MimyrConfig(**ckp["model_args"]).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Finetuned artifacts written to {args.output_dir}")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


def main():
    train(get_parser().parse_args())


if __name__ == "__main__":
    main()