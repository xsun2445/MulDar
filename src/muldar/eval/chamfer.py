import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict


def _ensure_2d(points: np.ndarray) -> np.ndarray:
    """Ensure points is a 2D array of shape (N, D)."""
    if points is None:
        raise ValueError("points must not be None")
    arr = np.asarray(points)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"points must be 2D (N, D); got shape {arr.shape}")
    if arr.size == 0 or arr.shape[0] == 0:
        raise ValueError("points must not be empty")
    return arr


def _nn_distances(points_src: np.ndarray, points_tgt: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances from each point in src to target set.

    Returns an array of shape (N_src,) with Euclidean distances.
    """
    src = _ensure_2d(points_src)
    tgt = _ensure_2d(points_tgt)
    tree = cKDTree(tgt)
    dists, _ = tree.query(src, k=1)
    return dists


def chamfer_components(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    squared: bool = True,
    reduction: str = "mean",
) -> Tuple[float, float]:
    """Compute directional Chamfer components A→B and B→A.

    - squared: if True, use squared Euclidean distances (common in literature)
    - reduction: one of {"mean", "sum", "none"}; controls aggregation of per-point distances

    Returns a tuple (a_to_b, b_to_a). If reduction == "none", returns the
    per-point arrays instead of scalars.
    """
    da = _nn_distances(points_a, points_b)
    db = _nn_distances(points_b, points_a)

    if squared:
        da = da ** 2
        db = db ** 2

    if reduction == "mean":
        return float(da.mean()), float(db.mean())
    if reduction == "sum":
        return float(da.sum()), float(db.sum())
    if reduction == "none":
        return da, db  # type: ignore[return-value]

    raise ValueError("reduction must be one of {'mean','sum','none'}")


def chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    squared: bool = True,
    reduction: str = "mean",
) -> float:
    """Symmetric Chamfer distance between two point clouds.

    Defined as reduction of NN distances A→B plus B→A.
    - squared: if True, uses squared distances
    - reduction: {"mean","sum"}. If you need per-direction arrays, use chamfer_components(..., reduction="none").
    """
    a2b, b2a = chamfer_components(points_a, points_b, squared=squared, reduction=reduction)
    if isinstance(a2b, np.ndarray) or isinstance(b2a, np.ndarray):
        raise ValueError("Use reduction != 'none' to get a scalar Chamfer distance")
    return float(a2b + b2a)


def precision_recall_fscore(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    *,
    tau: float,
    squared: bool = False,
) -> Tuple[float, float, float]:
    """Compute precision/recall/F-score at threshold tau using NN distances.

    - Precision: fraction of predicted points that match a GT within tau.
    - Recall: fraction of GT points that are covered by a prediction within tau.
    - F-score: harmonic mean of precision and recall (0 if both are 0).

    tau is applied on Euclidean distances; if squared=True, we compare with tau^2.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    pred_to_gt = _nn_distances(points_pred, points_gt)
    gt_to_pred = _nn_distances(points_gt, points_pred)

    if squared:
        thr = tau ** 2
        pred_hits = pred_to_gt ** 2 <= thr
        gt_hits = gt_to_pred ** 2 <= thr
    else:
        thr = tau
        pred_hits = pred_to_gt <= thr
        gt_hits = gt_to_pred <= thr

    precision = float(pred_hits.mean()) if pred_hits.size > 0 else 0.0
    recall = float(gt_hits.mean()) if gt_hits.size > 0 else 0.0
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = float(2 * precision * recall / (precision + recall))

    return precision, recall, fscore


def evaluate_point_clouds(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    *,
    tau: Optional[float] = None,
    squared: bool = True,
    reduction: str = "mean",
) -> Dict[str, float]:
    """Evaluate two point clouds with Chamfer distance and optional PR/F-score.

    Returns a dict with keys:
    - 'cd': symmetric Chamfer distance
    - 'cd_pred_to_gt': directional component (pred→gt)
    - 'cd_gt_to_pred': directional component (gt→pred)
    Optionally (if tau provided): 'precision', 'recall', 'fscore'.
    """
    cd_p2g, cd_g2p = chamfer_components(points_pred, points_gt, squared=squared, reduction=reduction)
    if isinstance(cd_p2g, np.ndarray) or isinstance(cd_g2p, np.ndarray):
        raise ValueError("evaluate_point_clouds expects reduction != 'none'")

    metrics: Dict[str, float] = {
        "cd": float(cd_p2g + cd_g2p),
        "cd_pred_to_gt": float(cd_p2g),
        "cd_gt_to_pred": float(cd_g2p),
    }

    if tau is not None:
        p, r, f = precision_recall_fscore(points_pred, points_gt, tau=tau, squared=False)
        metrics.update({
            "precision": p,
            "recall": r,
            "fscore": f,
        })

    return metrics
