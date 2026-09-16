import math
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm



def one_hot_encode(celltypes):
    unique_types = list(set(celltypes))
    I = np.eye(len(unique_types), dtype=np.float32)
    encoding_dict = {ctype: I[i] for i, ctype in enumerate(unique_types)}
    return encoding_dict, len(unique_types)


def soft_accuracy(
    gt_celltypes,
    gt_positions,
    pred_celltypes,
    pred_positions,
    radius=None,
    k=0,
    return_list=False,
    return_percent=False,
    sample=None,
):
    encoding_dict, num_classes = one_hot_encode(gt_celltypes + pred_celltypes)

    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]
    else:
        samples = gt_positions

    # Pre-compute encoding matrices for fast indexing
    gt_encoding_matrix = np.stack([encoding_dict[ct] for ct in gt_celltypes])    # (N_gt, C)
    pred_encoding_matrix = np.stack([encoding_dict[ct] for ct in pred_celltypes]) # (N_pred, C)

    if k > 0:
        # Fully vectorized kNN path: batch query all sample points at once
        _, gt_all_idx = gt_tree.query(samples, k=k + 1, workers=-1)   # (M, k+1)
        _, pred_all_idx = pred_tree.query(samples, k=k + 1, workers=-1)
        gt_sums = gt_encoding_matrix[gt_all_idx[:, 1:]].sum(axis=1)   # (M, C)
        pred_sums = pred_encoding_matrix[pred_all_idx[:, 1:]].sum(axis=1)

        gt_norms = np.linalg.norm(gt_sums, axis=1, keepdims=True)
        pred_norms = np.linalg.norm(pred_sums, axis=1, keepdims=True)
        gt_distributions = np.divide(gt_sums, gt_norms, out=np.zeros_like(gt_sums), where=gt_norms != 0)
        pred_distributions = np.divide(pred_sums, pred_norms, out=np.zeros_like(pred_sums), where=pred_norms != 0)
        result = (gt_distributions * pred_distributions).sum(axis=1).tolist()
    else:
        # Radius path: batch query with workers=-1, then loop with fast numpy ops
        gt_neighbors_all = gt_tree.query_ball_point(samples, radius, workers=-1)
        pred_neighbors_all = pred_tree.query_ball_point(samples, radius, workers=-1)

        gt_distributions = []
        pred_distributions = []
        result = []
        for i, (gt_neighbors, pred_neighbors) in enumerate(zip(gt_neighbors_all, pred_neighbors_all)):
            gt_encoding_sum = gt_encoding_matrix[gt_neighbors].sum(axis=0) if len(gt_neighbors) > 0 else np.zeros(num_classes)
            pred_encoding_sum = pred_encoding_matrix[pred_neighbors].sum(axis=0) if len(pred_neighbors) > 0 else np.zeros(num_classes)

            gt_norm = np.linalg.norm(gt_encoding_sum)
            pred_norm = np.linalg.norm(pred_encoding_sum)

            gt_distribution = gt_encoding_sum / gt_norm if gt_norm != 0 else np.zeros(num_classes)
            pred_distribution = pred_encoding_sum / pred_norm if pred_norm != 0 else np.zeros(num_classes)
            gt_distributions.append(gt_distribution)
            pred_distributions.append(pred_distribution)

            similarity = float(np.dot(gt_distribution, pred_distribution))
            result.append(similarity)

            if i % 10000 == 0:
                print(np.mean(result))

        gt_distributions = np.array(gt_distributions) if gt_distributions else np.zeros((0, num_classes))

    if return_percent:
        counts = np.sum([encoding_dict[ct] for ct in gt_celltypes], axis=0) / np.sum(
            [encoding_dict[ct] for ct in gt_celltypes]
        )
        return [(gd * counts).sum() for gd in gt_distributions]
    if return_list:
        return result

    return np.mean(result) if result else 0.0


def celltype_abundance_metrics(gt_tokens, pred_tokens, rare_frac=0.5):
    """Global cell-type abundance fidelity, with emphasis on the long tail.

    soft_accuracy / soft_spearman are dominated by the common classes (L2-norm +
    cosine), so they barely move when rare cell types collapse. These metrics
    target exactly that: does a rare type still appear at ~the right abundance?

    Returns a dict:
      tvd_all         : total-variation distance between gt and pred class
                        frequencies over ALL classes. 0 = identical, lower better.
      rare_tvd        : TVD restricted to GT-rare classes (bottom `rare_frac` of
                        GT-present classes by frequency). 0 = identical, lower better.
      rare_mass_ratio : (pred mass on GT-rare classes) / (gt mass on them). The
                        single most interpretable "do rare types appear at similar
                        abundance" number: 1.0 = matched, <1 = rare types under-
                        represented (the collapse signature), >1 = over-represented.
    """
    gt_tokens = np.asarray(gt_tokens).astype(int)
    pred_tokens = np.asarray(pred_tokens).astype(int)
    C = int(max(gt_tokens.max(), pred_tokens.max())) + 1

    gt_freq = np.bincount(gt_tokens, minlength=C).astype(np.float64)
    gt_freq /= gt_freq.sum()
    pred_freq = np.bincount(pred_tokens, minlength=C).astype(np.float64)
    pred_freq /= pred_freq.sum()

    tvd_all = float(0.5 * np.abs(gt_freq - pred_freq).sum())

    present = np.where(gt_freq > 0)[0]  # classes that actually occur in GT
    # rare = bottom `rare_frac` of GT-present classes by frequency
    cutoff = np.quantile(gt_freq[present], rare_frac)
    rare = present[gt_freq[present] <= cutoff]
    if len(rare) > 0:
        rare_tvd = float(0.5 * np.abs(gt_freq[rare] - pred_freq[rare]).sum())
        gt_rare_mass = gt_freq[rare].sum()
        rare_mass_ratio = float(pred_freq[rare].sum() / gt_rare_mass) if gt_rare_mass > 0 else 0.0
    else:
        rare_tvd = 0.0
        rare_mass_ratio = 1.0

    return {
        "tvd_all": tvd_all,
        "rare_tvd": rare_tvd,
        "rare_mass_ratio": rare_mass_ratio,
    }


def local_composition_entropy(gt_tokens, gt_positions, pred_tokens, pred_positions,
                              radius=0.05, sample=None):
    """Per-neighborhood cell-type diversity match.

    For each sample point, compare the Shannon entropy of the local cell-type
    composition (neighbors within `radius`) between GT and prediction. Mode
    collapse shows up as prediction entropy systematically BELOW GT entropy.

    Returns a dict:
      entropy_mae  : mean |H(gt nbhd) - H(pred nbhd)|, nats. 0 = identical diversity.
      entropy_bias : mean (H(pred) - H(gt)). Negative => predictions are less
                     diverse than GT (the collapse signature); ~0 = right diversity.
    """
    gt_tokens = np.asarray(gt_tokens).astype(int)
    pred_tokens = np.asarray(pred_tokens).astype(int)
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    if sample is not None:
        n = int(len(gt_positions) * sample / 100)
        idx = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    gt_nbrs = gt_tree.query_ball_point(samples, radius, workers=-1)
    pred_nbrs = pred_tree.query_ball_point(samples, radius, workers=-1)

    def _entropy(tokens):
        if len(tokens) == 0:
            return None
        counts = np.bincount(tokens)
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum())

    abs_diffs, signed_diffs = [], []
    for gn, pn in zip(gt_nbrs, pred_nbrs):
        hg = _entropy(gt_tokens[gn])
        hp = _entropy(pred_tokens[pn])
        if hg is None or hp is None:
            continue
        abs_diffs.append(abs(hp - hg))
        signed_diffs.append(hp - hg)

    return {
        "entropy_mae": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
        "entropy_bias": float(np.mean(signed_diffs)) if signed_diffs else 0.0,
    }



def composition_pairwise(gt_tokens, gt_positions, pred_tokens, pred_positions,
                         radius=0.05, rare_frac=0.5, sample=None, n_strata=4, seed=0):
    """PAIRWISE (same-location) rare-cell-type fidelity.

    The global rare_mass_ratio / tvd_all can look fine even if the model sprinkles
    rare types at a ~uniform base rate everywhere, while GT concentrates them in
    specific high-heterogeneity regions -- the two just have to agree on AVERAGE.
    This instead compares, at each sample location, the rare-cell FRACTION of the
    GT neighborhood vs the SAME location's PREDICTED neighborhood (1-to-1).

    "rare" = the bottom `rare_frac` of GT-present classes by global frequency.

    Returns a dict:
      rare_frac_pearson  : Pearson corr of (gt_rare_frac, pred_rare_frac) across
                           matched neighborhoods. LOW => rare cells are in the wrong
                           places (uniform base level), even if global abundance matches.
      rare_frac_spearman : rank version of the above.
      rare_frac_mae      : mean |gt_rare_frac - pred_rare_frac| per neighborhood.
      strata_gt / strata_pred : mean gt/pred rare-fraction within quartiles of
                           gt_rare_frac (low->high heterogeneity). If strata_pred is
                           ~flat while strata_gt rises, the model is NOT tracking the
                           spatial heterogeneity structure (the uniform-base failure).
    """
    gt_tokens = np.asarray(gt_tokens).astype(int)
    pred_tokens = np.asarray(pred_tokens).astype(int)
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)

    # rare set from GT global frequency
    C = int(max(gt_tokens.max(), pred_tokens.max())) + 1
    gt_freq = np.bincount(gt_tokens, minlength=C).astype(np.float64)
    gt_freq /= gt_freq.sum()
    present = np.where(gt_freq > 0)[0]
    cutoff = np.quantile(gt_freq[present], rare_frac)
    rare_mask = np.zeros(C, dtype=bool)
    rare_mask[present[gt_freq[present] <= cutoff]] = True
    gt_is_rare = rare_mask[gt_tokens]
    pred_is_rare = rare_mask[pred_tokens]

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    if sample is not None:
        rng = np.random.default_rng(seed)
        n = int(len(gt_positions) * sample / 100)
        idx = rng.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    gt_nbrs = gt_tree.query_ball_point(samples, radius, workers=-1)
    pred_nbrs = pred_tree.query_ball_point(samples, radius, workers=-1)

    gt_rf, pred_rf = [], []
    for gn, pn in zip(gt_nbrs, pred_nbrs):
        if len(gn) == 0 or len(pn) == 0:
            continue  # need both sides populated to compare 1-to-1
        gt_rf.append(gt_is_rare[gn].mean())
        pred_rf.append(pred_is_rare[pn].mean())
    gt_rf = np.asarray(gt_rf)
    pred_rf = np.asarray(pred_rf)

    out = {"rare_frac_pearson": 0.0, "rare_frac_spearman": 0.0,
           "rare_frac_mae": 0.0, "strata_gt": [], "strata_pred": []}
    if len(gt_rf) < 2:
        return out
    out["rare_frac_mae"] = float(np.abs(gt_rf - pred_rf).mean())
    if gt_rf.std() > 0 and pred_rf.std() > 0:
        out["rare_frac_pearson"] = float(pearsonr(gt_rf, pred_rf)[0])
        out["rare_frac_spearman"] = float(spearmanr(gt_rf, pred_rf).correlation)

    # heterogeneity strata: quantile bins of gt_rare_frac
    edges = np.quantile(gt_rf, np.linspace(0, 1, n_strata + 1))
    edges[-1] = np.inf
    bins = np.digitize(gt_rf, edges[1:-1])
    for b in range(n_strata):
        m = bins == b
        out["strata_gt"].append(float(gt_rf[m].mean()) if m.any() else float("nan"))
        out["strata_pred"].append(float(pred_rf[m].mean()) if m.any() else float("nan"))
    return out


def delauney_colocalization(
    gt_celltypes, gt_positions, pred_celltypes, pred_positions, encoding_dict=None
):
    """
    Build Delaunay graph for GT and Pred positions and compute L1 distance
    between their edge-type count maps (sparse via Counter).

    Parameters
    ----------
    gt_celltypes : list[str]
        Ground-truth cell type labels (length N).
    gt_positions : array-like, shape (N, 2)
        Ground-truth positions.
    pred_celltypes : list[str]
        Predicted cell type labels (length M).
    pred_positions : array-like, shape (M, 2)
        Predicted positions.
    encoding_dict : dict, optional
        Mapping from cell type to index. If None, builds internally.

    Returns
    -------
    l1_distance : float
        L1 distance (sum of absolute differences) between GT and Pred edge-type counts.
    """

    # Build encoding dict if not given
    if encoding_dict is None:
        all_cts = list(set(gt_celltypes) | set(pred_celltypes))
        encoding_dict = {ct: i for i, ct in enumerate(all_cts)}

    def build_count_counter(celltypes, positions):
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
                edges.add((a, b))

        counter = Counter()
        for a, b in edges:
            ia, ib = encoding_dict[celltypes[a]], encoding_dict[celltypes[b]]
            counter[(ia, ib)] += 1
            if ia != ib:
                counter[(ib, ia)] += 1  # keep symmetric
        return counter

    gt_counter = build_count_counter(gt_celltypes, np.array(gt_positions))
    pred_counter = build_count_counter(pred_celltypes, np.array(pred_positions))

    all_keys = set(gt_counter.keys()) | set(pred_counter.keys())
    l1_distance = sum(abs(gt_counter[k] - pred_counter[k]) for k in all_keys)

    return l1_distance / len(gt_positions)


def gridized_l1_distance(
    gt_positions, pred_positions, radius=None, k=0, grid_size=50, return_list=False
):
    """
    Estimate weighted L1 distance between two distributions by evaluating
    density differences on a regular grid of points spanning the data space.

    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth positions.
    pred_positions : array-like, shape (n_pred, d)
        Predicted positions.
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point distances. Otherwise return mean.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d / 2) / math.gamma(d / 2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    if k > 0:
        # kNN mode: batch query all grid points at once
        gt_d, _ = gt_tree.query(grid_points, k=k, workers=-1)    # (N_grid, k)
        pred_d, _ = pred_tree.query(grid_points, k=k, workers=-1)
        r_gt = gt_d[:, -1]
        r_pred = pred_d[:, -1]
        p_hat = np.where(r_gt > 0, k / (n_gt * Vd * r_gt**d), 0.0)
        q_hat = np.where(r_pred > 0, k / (n_pred * Vd * r_pred**d), 0.0)
    else:
        # Radius mode: single parallel batch query, return_length avoids list-of-lists
        k_gt = gt_tree.query_ball_point(grid_points, radius, workers=-1, return_length=True).astype(float)
        k_pred = pred_tree.query_ball_point(grid_points, radius, workers=-1, return_length=True).astype(float)
        p_hat = k_gt / (n_gt * Vd * radius**d) if radius > 0 else np.zeros(len(grid_points))
        q_hat = k_pred / (n_pred * Vd * radius**d) if radius > 0 else np.zeros(len(grid_points))

    results = np.abs(p_hat - q_hat)
    return results.tolist() if return_list else float(np.mean(results)) if len(results) > 0 else 0.0



def gridized_kl_divergence(
    gt_positions,
    pred_positions,
    radius=None,
    k=0,
    grid_size=50,
    return_list=False,
    eps=1e-200,
):
    """
    Estimate KL divergence KL(P||Q) between two distributions by evaluating
    density estimates on a regular grid of points spanning the data space.

    Parameters
    ----------
    gt_positions : array-like, shape (n_gt, d)
        Ground-truth samples (distribution P).
    pred_positions : array-like, shape (n_pred, d)
        Predicted samples (distribution Q).
    radius : float, optional
        Radius for density estimation (used if k == 0).
    k : int, optional
        k for kNN density estimation. If > 0, kNN mode is used.
    grid_size : int, optional
        Number of grid points per dimension.
    return_list : bool, optional
        If True, return list of per-grid-point KL terms. Otherwise return mean.
    eps : float, optional
        Small constant to avoid log(0) and division by zero.
    """
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)

    d = gt_positions.shape[1]
    Vd = np.pi ** (d / 2) / math.gamma(d / 2 + 1)  # volume of unit d-ball
    n_gt, n_pred = len(gt_positions), len(pred_positions)

    # Build trees
    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # Grid bounding box
    mins = np.minimum(gt_positions.min(axis=0), pred_positions.min(axis=0))
    maxs = np.maximum(gt_positions.max(axis=0), pred_positions.max(axis=0))
    grid_axes = [np.linspace(mins[i], maxs[i], grid_size) for i in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_points = np.stack([m.ravel() for m in mesh], axis=-1)

    if k > 0:
        # kNN mode: batch query all grid points at once
        gt_d, _ = gt_tree.query(grid_points, k=k, workers=-1)    # (N_grid, k)
        pred_d, _ = pred_tree.query(grid_points, k=k, workers=-1)
        r_gt = gt_d[:, -1]
        r_pred = pred_d[:, -1]
        p_hat = np.where(r_gt > 0, k / (n_gt * Vd * r_gt**d), 0.0)
        q_hat = np.where(r_pred > 0, k / (n_pred * Vd * r_pred**d), 0.0)
    else:
        # Radius mode: single parallel batch query
        k_gt = gt_tree.query_ball_point(grid_points, radius, workers=-1, return_length=True).astype(float)
        k_pred = pred_tree.query_ball_point(grid_points, radius, workers=-1, return_length=True).astype(float)
        p_hat = k_gt / (n_gt * Vd * radius**d) if radius > 0 else np.zeros(len(grid_points))
        q_hat = k_pred / (n_pred * Vd * radius**d) if radius > 0 else np.zeros(len(grid_points))

    p_hat = np.maximum(p_hat, eps)
    q_hat = np.maximum(q_hat, eps)
    results = p_hat * np.log(p_hat / q_hat)
    return results.tolist() if return_list else float(np.mean(results)) if len(results) > 0 else 0.0


import numpy as np
import scipy.sparse as sp


def _nnz_per_gene(X):
    if sp.issparse(X):
        return np.asarray(X.getnnz(axis=0)).ravel()
    return np.asarray((X > 0).sum(axis=0)).ravel()


def intersect_and_filter_X(gt_adata, pred_adata, min_expr_cells=0, gene_set=None, filter_by_gt=False):
    # 1) intersect genes (order is preserved by AnnData slicing)
    common_genes = gt_adata.var_names.intersection(pred_adata.var_names)

    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between gt_adata and pred_adata.")

    # 2) if a gene_set is provided, restrict to those genes within common_genes.
    #    NB: this only narrows the candidate genes -- the expression filter below
    #    (including filter_by_gt) still runs, so --metric_filter_by_gt is honored
    #    whether or not a gene_set was supplied.
    if gene_set is not None:
        common_genes = common_genes.intersection(np.asarray(gene_set))
        if len(common_genes) == 0:
            raise ValueError("No genes in gene_set overlap with common genes.")

    gt_common = gt_adata[:, common_genes]
    pred_common = pred_adata[:, common_genes]

    # 3) require expression in each adata (or only in gt when filter_by_gt=True)
    gt_nnz = _nnz_per_gene(gt_common.X)
    if filter_by_gt:
        keep_mask = gt_nnz >= min_expr_cells
    else:
        pred_nnz = _nnz_per_gene(pred_common.X)
        keep_mask = (gt_nnz >= min_expr_cells) & (pred_nnz >= min_expr_cells)

    if not np.any(keep_mask):
        raise ValueError("No genes pass the expression filter.")

    # 3) return filtered .X matrices and the kept gene names (same order)
    gt_X_filtered = gt_common[:, keep_mask].X
    pred_X_filtered = pred_common[:, keep_mask].X
    kept_genes = common_genes[keep_mask]
    if sp.issparse(gt_X_filtered):
        gt_X_filtered = gt_X_filtered.todense()
    if sp.issparse(pred_X_filtered):
        pred_X_filtered = pred_X_filtered.todense()

    return gt_X_filtered.tolist(), pred_X_filtered.tolist(), kept_genes




def soft_correlation(
    gt_adata,
    gt_positions,
    pred_adata,
    pred_positions,
    radius=None,
    k=0,
    sample=None,
    return_list=False,
    corr_type="pearson",
    gene_set=None,
    filter_by_gt=False,
):
    """
    gt_expressions, pred_expressions: list or array of gene expression vectors (shape [num_cells, num_genes])
    gt_positions, pred_positions: list or array of positions (shape [num_cells, 2] or [num_cells, 3])
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """

    if corr_type == "pearson":
        corr_fn = pearsonr
    elif corr_type == "spearman":
        corr_fn = spearmanr
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(
        gt_adata, pred_adata, 1, gene_set, filter_by_gt=filter_by_gt
    )
    print(f"running soft_correlation on {len(genes)} genes")
    gt_positions = np.array(gt_positions)
    pred_positions = np.array(pred_positions)
    gt_expressions = np.array(gt_expressions)
    pred_expressions = np.array(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    if sample is not None:
        percent = sample
        n = int(len(gt_positions) * percent / 100)
        indices = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[indices]
    else:
        samples = gt_positions

    if k > 0:
        # Fully vectorized kNN path
        _, gt_all_idx = gt_tree.query(samples, k=k + 1, workers=-1)   # (M, k+1)
        _, pred_all_idx = pred_tree.query(samples, k=k + 1, workers=-1)
        gt_sums_mat = gt_expressions[gt_all_idx[:, 1:]].sum(axis=1)   # (M, G)
        pred_sums_mat = pred_expressions[pred_all_idx[:, 1:]].sum(axis=1)

        if return_list:
            correlations_all = [corr_fn(gt_sums_mat[i], pred_sums_mat[i])[0] for i in range(len(samples))]
            return correlations_all

        gt_sums = gt_sums_mat.flatten()
        pred_sums = pred_sums_mat.flatten()
    else:
        # Radius path: batch query with workers=-1, loop only for variable-length sums
        gt_neighbors_all = gt_tree.query_ball_point(samples, radius, workers=-1)
        pred_neighbors_all = pred_tree.query_ball_point(samples, radius, workers=-1)

        gt_sums_list = []
        pred_sums_list = []
        correlations_all = []
        for i, (gt_nbrs, pred_nbrs) in enumerate(zip(gt_neighbors_all, pred_neighbors_all)):
            gt_sum = gt_expressions[gt_nbrs].sum(axis=0) if len(gt_nbrs) > 0 else np.zeros(gt_expressions.shape[1])
            pred_sum = pred_expressions[pred_nbrs].sum(axis=0) if len(pred_nbrs) > 0 else np.zeros(pred_expressions.shape[1])
            gt_sums_list.append(gt_sum)
            pred_sums_list.append(pred_sum)
            if return_list:
                pred_sum[0] = pred_sum[0] + 1e-15
                correlations_all.append(corr_fn(gt_sum, pred_sum)[0])

        if return_list:
            return correlations_all

        gt_sums = np.array(gt_sums_list).flatten() if gt_sums_list else np.array([])
        pred_sums = np.array(pred_sums_list).flatten() if pred_sums_list else np.array([])

    if len(gt_sums) == 0 or len(pred_sums) == 0:
        return 0.0

    # Avoid NaN when pred is all zeros
    pred_sums = pred_sums.copy()
    pred_sums[0] = pred_sums[0] + 1e-15
    correlation, _ = corr_fn(gt_sums, pred_sums)
    return correlation


def soft_correlation_top_100(
    gt_adata,
    gt_positions,
    pred_adata,
    pred_positions,
    radius=None,
    k=0,
    sample=None,
    return_list=False,
    corr_type="pearson",
):
    """
    Wrapper around soft_correlation that restricts to the top 100 genes
    by total expression in the ground truth (gt_adata).
    """
    import scipy.sparse as sp

    X = gt_adata.X
    if sp.issparse(X):
        gene_totals = np.asarray(X.sum(axis=0)).ravel()
    else:
        gene_totals = np.asarray(X).sum(axis=0).ravel()

    top_100_idx = np.argsort(gene_totals)[-100:]
    top_100_genes = gt_adata.var_names[top_100_idx]

    return soft_correlation(
        gt_adata,
        gt_positions,
        pred_adata,
        pred_positions,
        radius=radius,
        k=k,
        sample=sample,
        return_list=return_list,
        corr_type=corr_type,
        gene_set=top_100_genes,
    )


def soft_f1(
    gt_adata,
    gt_positions,
    pred_adata,
    pred_positions,
    radius=None,
    k=0,
    sample=None,
    return_list=False,
    gene_set=None,
    filter_by_gt=False,
):
    """
    gt_expressions, pred_expressions: array of shape [num_cells, num_genes]
    gt_positions, pred_positions: array of shape [num_cells, 2] or [num_cells, 3]
    radius: radius for neighbor search (if k=0)
    k: number of neighbors to consider (if k>0)
    sample: if provided, percentage of gt_positions to sample
    """
    gt_expressions, pred_expressions, genes = intersect_and_filter_X(
        gt_adata, pred_adata, gene_set=gene_set, filter_by_gt=filter_by_gt
    )
    gt_positions = np.asarray(gt_positions)
    pred_positions = np.asarray(pred_positions)
    gt_expressions = np.asarray(gt_expressions)
    pred_expressions = np.asarray(pred_expressions)

    gt_tree = cKDTree(gt_positions)
    pred_tree = cKDTree(pred_positions)

    # sampling
    if sample is not None:
        n = int(len(gt_positions) * sample / 100)
        idx = np.random.choice(len(gt_positions), size=n, replace=False)
        samples = gt_positions[idx]
    else:
        samples = gt_positions

    if k > 0:
        # Fully vectorized kNN path
        _, gt_all_idx = gt_tree.query(samples, k=k + 1, workers=-1)   # (M, k+1)
        _, pred_all_idx = pred_tree.query(samples, k=k + 1, workers=-1)
        gt_sums = gt_expressions[gt_all_idx[:, 1:]].sum(axis=1)       # (M, G)
        pred_sums = pred_expressions[pred_all_idx[:, 1:]].sum(axis=1)

        pred_pos = pred_sums > 0                                        # (M, G) bool
        gt_pos = gt_sums > 0
        n_pred_pos = pred_pos.sum(axis=1).astype(float)                # (M,)
        n_gt_pos = gt_pos.sum(axis=1).astype(float)
        true_pos = np.logical_and(pred_pos, gt_pos).sum(axis=1).astype(float)

        precisions_arr = np.where(n_pred_pos > 0, true_pos / n_pred_pos, 0.0)
        recalls_arr = np.where(n_gt_pos > 0, true_pos / n_gt_pos, 0.0)
        denom = precisions_arr + recalls_arr
        f1s_arr = np.where(denom > 0, 2 * precisions_arr * recalls_arr / denom, 0.0)

        if return_list:
            return f1s_arr.tolist()
        return (float(np.mean(f1s_arr)), float(np.mean(precisions_arr)), float(np.mean(recalls_arr)))
    else:
        # Radius path: batch query with workers=-1
        gt_neighbors_all = gt_tree.query_ball_point(samples, radius, workers=-1)
        pred_neighbors_all = pred_tree.query_ball_point(samples, radius, workers=-1)

        precisions = []
        recalls = []
        f1s = []
        for i, (gt_nbrs, pred_nbrs) in enumerate(zip(gt_neighbors_all, pred_neighbors_all)):
            gt_sum = gt_expressions[gt_nbrs].sum(axis=0) if len(gt_nbrs) > 0 else np.zeros(gt_expressions.shape[1])
            pred_sum = pred_expressions[pred_nbrs].sum(axis=0) if len(pred_nbrs) > 0 else np.zeros(pred_expressions.shape[1])

            pred_pos = pred_sum > 0
            n_pred_pos = pred_pos.sum()
            if n_pred_pos > 0:
                true_pos = np.logical_and(pred_pos, gt_sum > 0).sum()
                p = true_pos / n_pred_pos
                n_gt_pos = (gt_sum > 0).sum()
                r = true_pos / n_gt_pos if n_gt_pos > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            else:
                p, r, f1 = 0.0, 0.0, 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)


        if return_list:
            return f1s
        return (
            float(np.mean(f1s)) if f1s else 0.0,
            float(np.mean(precisions)) if precisions else 0.0,
            float(np.mean(recalls)) if recalls else 0.0,
        )
