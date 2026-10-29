"""Hiệu chỉnh doc_confidence (kế hoạch mục 5): độ tin cậy thô của luật -> xác suất đúng thực nghiệm, rồi chọn ngưỡng bằng risk-coverage.

Mỗi trường bắt buộc (tên người bán, ngày, tổng) có một hồi quy logistic nhỏ trên tín hiệu thô (độ tin cậy luật, rec_score, số ứng viên đồng ý, ...).
`doc_confidence` = min xác suất đã hiệu chỉnh theo các trường bắt buộc. Hồi quy tự cài bằng numpy (IRLS + L2) để không thêm phụ thuộc.
Nhãn đúng/sai và nhóm merchant lấy từ `score_page` (src/pipelines/receipt_flow.py); ước lượng chất lượng dùng cross-validation theo nhóm.
"""
import json
from dataclasses import dataclass
from itertools import pairwise
from math import log
from pathlib import Path

import numpy as np

FIELDS = ("merchant_name", "date", "total")        # ánh xạ từ REQUIRED_FIELDS ở receipt_flow
LABEL_KEY = {"merchant_name": "seller", "date": "date", "total": "total"}   # cột nhãn tương ứng trong score_page
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "doc_confidence_calibration.json"
TOTAL_RULES = ("keyword:same_line", "keyword:same_row", "keyword:next_row")     # 'fallback:max_amount' là mức tham chiếu


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-3), 1 - 1e-3)
    return log(p / (1 - p))


def feature_names(name: str) -> list[str]:
    base = ["logit_conf", "ocr_conf", "mean_rec"]
    extra = {"total": [*TOTAL_RULES, "agree2", "is_max", "distinct_gt1", "strong_key"],
             "date": ["keyword", "n_distinct_gt1", "agree2"],
             "merchant_name": ["brand", "first_line", "index"]}[name]
    return base + extra


def field_vector(name: str, fd: dict, mean_rec: float | None) -> list[float]:
    """fd = một mục của result['fields'] (confidence, rule, features)."""
    f = fd.get("features") or {}
    vec = [_logit(fd["confidence"]), f.get("ocr_conf", 0.0), mean_rec if mean_rec is not None else 0.5]
    if name == "total":
        vec += [float(fd["rule"] == r) for r in TOTAL_RULES]
        vec += [float(f.get("agree", 0) >= 2), f.get("is_max", 0.0), float(f.get("distinct", 0) > 1), float(f.get("key_weight", 0) >= 3)]
    elif name == "date":
        vec += [f.get("keyword", 0.0), float(f.get("n_distinct", 0) > 1), float(f.get("agree", 0) >= 2)]
    else:
        vec += [f.get("brand", 0.0), float(f.get("index", 9) == 0), float(f.get("index", 9))]
    return vec


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, l2: float = 1.0, iters: int = 50) -> dict:
    """Logistic có trọng số + L2 (không phạt hệ số chặn) trên đặc trưng đã chuẩn hoá, giải bằng IRLS."""
    n, d = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-9] = 1.0
    Z = np.hstack([np.ones((n, 1)), (X - mean) / std])
    beta = np.zeros(d + 1)
    if len(set(y.tolist())) < 2:   # một lớp duy nhất: xác suất hằng
        beta[0] = _logit((y.mean() * w.sum() + 0.5) / (w.sum() + 1))
        return {"mean": mean.tolist(), "std": std.tolist(), "coef": [0.0] * d, "intercept": float(beta[0])}
    reg = l2 * np.eye(d + 1)
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) + reg @ beta
        H = Z.T @ (Z * (w * p * (1 - p))[:, None]) + reg + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": beta[1:].tolist(), "intercept": float(beta[0])}


def predict_logistic(model: dict, X: np.ndarray) -> np.ndarray:
    z = ((np.atleast_2d(X) - np.asarray(model["mean"])) / np.asarray(model["std"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Calibrator:
    models: dict[str, dict]
    meta: dict

    def is_in_sample(self, work: Path) -> bool:
        """Metadata mới dùng đường dẫn tuyệt đối; file cũ chỉ có tên run thì đối chiếu tên."""
        if "run_paths" in self.meta:
            return str(work.resolve()) in self.meta["run_paths"]
        return work.name in self.meta.get("runs", [])

    def field_probs(self, fields: dict, mean_rec: float | None) -> dict[str, float]:
        """fields = result['fields']; trường thiếu giá trị -> 0."""
        out = {}
        for name in FIELDS:
            fd = fields[name]
            out[name] = 0.0 if fd["value"] is None else float(predict_logistic(self.models[name], np.array(field_vector(name, fd, mean_rec)))[0])
        return out

    def doc_confidence(self, fields: dict, mean_rec: float | None) -> float:
        return min(self.field_probs(fields, mean_rec).values())

    def to_json(self) -> dict:
        return {"meta": self.meta, "models": {k: {**m, "features": feature_names(k)} for k, m in self.models.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "Calibrator":
        models = {k: {kk: vv for kk, vv in m.items() if kk != "features"} for k, m in data["models"].items()}
        for k, m in data["models"].items():
            if m["features"] != feature_names(k):
                raise ValueError(f"Calibration features out of date for {k}; refit with scripts/calibrate_confidence.py")
        return cls(models, data["meta"])


def load_calibrator(path: Path | None = None) -> Calibrator | None:
    path = path or DEFAULT_PATH
    return Calibrator.from_json(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


# ---------- Dữ liệu hiệu chỉnh và đánh giá ----------
def group_weights(groups: list[str]) -> np.ndarray:
    """Mỗi nhóm merchant có tổng trọng số như nhau (mục tiêu chính là template chưa thấy), chuẩn hoá để trung bình bằng 1."""
    counts = {g: groups.count(g) for g in set(groups)}
    w = np.array([1.0 / counts[g] for g in groups])
    return w * len(w) / w.sum()


def fit_calibrator(samples: list[dict], l2: float = 1.0, weighted: bool = True) -> Calibrator:
    """samples: mỗi phần tử {fields, mean_rec, group_id, labels: {seller,date,total}} (nhãn None = không có tham chiếu)."""
    models = {}
    for name in FIELDS:
        rows = [s for s in samples if s["fields"][name]["value"] is not None and s["labels"][LABEL_KEY[name]] is not None]
        X = np.array([field_vector(name, s["fields"][name], s["mean_rec"]) for s in rows])
        y = np.array([s["labels"][LABEL_KEY[name]] for s in rows], dtype=float)
        w = group_weights([s["group_id"] for s in rows]) if weighted else None
        models[name] = fit_logistic(X, y, w, l2)
    return Calibrator(models, {"documents": len(samples), "groups": len({s["group_id"] for s in samples}), "l2": l2, "weighted": weighted})


def grouped_oof(samples: list[dict], folds: int = 10, repeats: int = 5, seed: int = 42, **kw) -> np.ndarray:
    """Xác suất từng trường out-of-fold, shape (số ảnh, len(FIELDS)): cross-validation chia theo nhóm merchant (không ảnh nào được
    dự đoán bởi mô hình đã thấy nhóm của nó), trung bình qua `repeats` lần chia khác nhau."""
    groups = sorted({s["group_id"] for s in samples})
    acc = np.zeros((len(samples), len(FIELDS)))
    for r in range(repeats):
        order = np.random.default_rng(seed + r).permutation(len(groups))
        fold_of = {groups[i]: k % folds for k, i in enumerate(order)}
        for f in range(folds):
            test = [i for i, s in enumerate(samples) if fold_of[s["group_id"]] == f]
            if not test:
                continue
            cal = fit_calibrator([s for s in samples if fold_of[s["group_id"]] != f], **kw)
            for i in test:
                probs = cal.field_probs(samples[i]["fields"], samples[i]["mean_rec"])
                acc[i] += [probs[name] for name in FIELDS]
    return acc / repeats


def auroc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    diff = score[pos][:, None] - score[neg][None, :]
    ww = w[pos][:, None] * w[neg][None, :]
    return float(((diff > 0) * ww + 0.5 * (diff == 0) * ww).sum() / ww.sum())


def calibration_metrics(prob: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 5) -> dict[str, float]:
    p = np.clip(prob, 1e-4, 1 - 1e-4)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (p >= lo) & (p <= hi) if hi == edges[-1] else (p >= lo) & (p < hi)
        if m.any():
            ece += w[m].sum() / w.sum() * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return {"brier": float(np.average((p - y) ** 2, weights=w)), "log_loss": float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w)),
            "auroc": auroc(p, y, w), "ece": float(ece)}


def risk_coverage(conf: np.ndarray, correct: np.ndarray, w: np.ndarray, taus) -> list[dict]:
    out = []
    for t in taus:
        m = conf >= t
        out.append({"tau": float(t), "documents": int(m.sum()), "coverage": float(w[m].sum() / w.sum()),
                    "error_rate": float(1 - np.average(correct[m], weights=w[m])) if m.any() else None})
    return out


def choose_tau_high(conf: np.ndarray, correct: np.ndarray, w: np.ndarray, target_error: float, min_docs: int = 15,
                    tau_low: float | None = None) -> float | None:
    """Ngưỡng nhỏ nhất đạt mục tiêu lỗi và min_docs; nếu có tau_low thì phải lớn hơn để các nhóm routing không chồng nhau."""
    for t in sorted({float(c) for c in conf}):
        if tau_low is not None and t <= tau_low:
            continue
        m = conf >= t
        if m.sum() >= min_docs and 1 - np.average(correct[m], weights=w[m]) <= target_error:
            return t
    return None
