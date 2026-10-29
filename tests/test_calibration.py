import json
from datetime import date

import numpy as np
import pytest

from src.config import get_pipeline_config
from src.ocr.types import OcrLine
from src.pipelines.calibration import (
    FIELDS,
    Calibrator,
    choose_tau_high,
    fit_calibrator,
    fit_logistic,
    group_weights,
    grouped_oof,
    predict_logistic,
    risk_coverage,
)
from src.pipelines.receipt_flow import process_page, required_keys

TODAY = date(2026, 10, 10)


def make_sample(i, group, good, conf):
    """Ảnh tổng hợp: độ tin cậy thô cao <-> thường đúng. good = nhãn đúng cho cả ba trường."""
    fields = {
        "merchant_name": {"value": "X", "confidence": conf, "rule": "top_lines", "features": {"ocr_conf": conf, "brand": 1.0, "index": 0}},
        "date": {"value": "2020-08-14", "confidence": conf, "rule": "date_regex", "features": {"ocr_conf": conf, "keyword": 1.0, "n_distinct": 1, "agree": 1}},
        "total": {"value": 72000, "confidence": conf, "rule": "keyword:same_row",
                  "features": {"ocr_conf": conf, "agree": 2, "key_weight": 3, "is_max": 1.0, "n_cands": 2, "distinct": 1}},
    }
    return {"image_id": f"{i}.jpg", "group_id": group, "fields": fields, "mean_rec": conf,
            "labels": {"seller": good, "date": good, "total": good}}


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        conf = float(rng.uniform(0.3, 0.95))
        out.append(make_sample(i, f"g{i % 20}", int(rng.random() < conf), conf))
    return out


def test_fit_logistic_is_monotone_and_handles_single_class():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 1))
    y = (rng.random(300) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
    model = fit_logistic(x, y)
    p = predict_logistic(model, np.array([[-2.0], [0.0], [2.0]]))
    assert p[0] < p[1] < p[2] and model["coef"][0] > 0
    assert predict_logistic(fit_logistic(x, np.ones(300)), np.array([[0.0]]))[0] > 0.9


def test_group_weights_equalize_groups():
    w = group_weights(["a", "a", "a", "a", "b"])
    assert w.mean() == pytest.approx(1.0)
    assert w[:4].sum() == pytest.approx(w[4])


def test_calibrator_orders_by_correctness_and_roundtrips():
    cal = fit_calibrator(synthetic())
    lo, hi = make_sample(0, "g0", 0, 0.35), make_sample(1, "g1", 1, 0.92)
    assert cal.doc_confidence(lo["fields"], lo["mean_rec"]) < cal.doc_confidence(hi["fields"], hi["mean_rec"])
    again = Calibrator.from_json(json.loads(json.dumps(cal.to_json())))
    assert again.doc_confidence(hi["fields"], hi["mean_rec"]) == pytest.approx(cal.doc_confidence(hi["fields"], hi["mean_rec"]))
    broken = cal.to_json()
    broken["models"]["total"]["features"] = ["old"]
    with pytest.raises(ValueError, match="out of date"):
        Calibrator.from_json(broken)


def test_missing_field_gets_zero_probability():
    cal = fit_calibrator(synthetic())
    s = make_sample(0, "g0", 1, 0.9)
    s["fields"]["date"]["value"] = None
    assert cal.field_probs(s["fields"], 0.9)["date"] == 0.0 and cal.doc_confidence(s["fields"], 0.9) == 0.0


def test_grouped_oof_shape_range_and_determinism():
    samples = synthetic(80)
    a, b = grouped_oof(samples, folds=4, repeats=2), grouped_oof(samples, folds=4, repeats=2)
    assert a.shape == (80, len(FIELDS)) and (a >= 0).all() and (a <= 1).all() and np.array_equal(a, b)


def test_choose_tau_high_respects_target_and_min_docs():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    ok = np.array([1, 1, 1, 0, 1, 0], float)
    w = np.ones(6)
    assert choose_tau_high(conf, ok, w, target_error=0.0, min_docs=2) == 0.7     # 0.9, 0.8, 0.7 đều đúng; thêm 0.6 sẽ có lỗi
    assert choose_tau_high(conf, ok, w, 0.0, 4) is None                          # 4 ảnh đầu đã có một lỗi
    assert choose_tau_high(conf, ok, w, 0.5, 1) == 0.4
    rc = risk_coverage(conf, ok, w, (0.7, 0.95))
    assert rc[0]["documents"] == 3 and rc[0]["error_rate"] == 0.0 and rc[1]["error_rate"] is None


def test_process_page_uses_calibrated_min_over_required_fields():
    cfg = get_pipeline_config()
    cal = fit_calibrator(synthetic())
    lines = [OcrLine(polygon=[[x, y], [x + 200, y], [x + 200, y + 24], [x, y + 24]], det_score=0.9, text=t, rec_score=0.93)
             for t, x, y in (("SIEU THI BACH HOA", 230, 120), ("Ngày: 14-08-2020", 200, 320), ("Tổng cộng:", 187, 920), ("72,000", 626, 912))]
    out = process_page({"image_id": "a.jpg", "lines": [ln.to_dict() for ln in lines]}, cfg, TODAY, cal)
    assert out["doc_confidence"] == pytest.approx(min(out["field_probs"][k] for k in required_keys(cfg)), abs=1e-4)
    assert out["doc_confidence_raw"] > 0 and "features" in out["fields"]["total"]
    assert process_page({"image_id": "a.jpg", "lines": [ln.to_dict() for ln in lines]}, cfg, TODAY)["field_probs"] is None


def test_required_keys_follow_config_and_skip_items():
    assert required_keys(get_pipeline_config()) == ["merchant_name", "date", "total"]


def test_a8_recommended_high_threshold_stays_above_configured_low():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct, weights = np.ones(4), np.ones(4)
    assert choose_tau_high(conf, correct, weights, 0.15, min_docs=2, tau_low=0.7) == 0.8
    assert choose_tau_high(conf, correct, weights, 0.15, min_docs=3, tau_low=0.7) is None
