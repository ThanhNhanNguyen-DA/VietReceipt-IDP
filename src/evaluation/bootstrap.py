"""Bootstrap theo nhóm (merchant/template) cho các chỉ số dạng tỉ số tổng (CER/WER) — kế hoạch mục 3.1: CI 95%, 1.000 lần, lấy mẫu lại theo merchant.

Mẫu trong cùng một nhóm không độc lập (cùng template), nên lấy mẫu lại cả nhóm chứ không lấy mẫu lại từng ảnh.
"""
from collections.abc import Sequence

import numpy as np


def _validate(groups, num, den, n_resamples, confidence):
    if not len(groups) or len(groups) != len(num) or len(groups) != len(den):
        raise ValueError("groups, num, den must be non-empty and have equal lengths")
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence must be in (0, 1)")
    if not np.isfinite(num).all() or not np.isfinite(den).all() or np.any(np.asarray(den) < 0) or sum(den) <= 0:
        raise ValueError("Finite counts and a positive total denominator are required")


def _group_sums(groups: Sequence[str], num: Sequence[float], den: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    uniq = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    n, d = np.zeros(len(uniq)), np.zeros(len(uniq))
    for g, a, b in zip(groups, num, den, strict=True):
        n[uniq[g]] += a
        d[uniq[g]] += b
    return n, d


def ratio_ci(groups, num, den, n_resamples=1000, confidence=0.95, seed=0) -> dict[str, float]:
    """CI của tổng(num)/tổng(den) khi lấy mẫu lại theo nhóm. -> {value, low, high, n_groups}."""
    _validate(groups, num, den, n_resamples, confidence)
    n, d = _group_sums(groups, num, den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_resamples, len(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = n[idx].sum(1) / d[idx].sum(1)
    a = (1 - confidence) / 2
    return {"value": float(n.sum() / d.sum()), "low": float(np.nanquantile(stats, a)), "high": float(np.nanquantile(stats, 1 - a)),
            "n_groups": len(n)}


def paired_ratio_diff_ci(groups, num_a, den_a, num_b, den_b, n_resamples=1000, confidence=0.95, seed=0) -> dict[str, float]:
    """CI của (tỉ số A - tỉ số B) trên CÙNG các mẫu lại: tách chênh lệch thật giữa hai cấu hình khỏi nhiễu chọn mẫu."""
    _validate(groups, num_a, den_a, n_resamples, confidence)
    _validate(groups, num_b, den_b, n_resamples, confidence)
    na, da = _group_sums(groups, num_a, den_a)
    nb, db = _group_sums(groups, num_b, den_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(na), size=(n_resamples, len(na)))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = na[idx].sum(1) / da[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)
    a = (1 - confidence) / 2
    return {"diff": float(na.sum() / da.sum() - nb.sum() / db.sum()), "low": float(np.nanquantile(stats, a)),
            "high": float(np.nanquantile(stats, 1 - a)), "p_a_better": float(np.mean(stats[np.isfinite(stats)] < 0))}
