"""Hồi quy A8: hiệu chỉnh không ghi đè run, lấy tau_low từ config và cảnh báo in-sample."""
import json
import sys
from datetime import date

import pytest

from scripts import calibrate_confidence, run_receipt_flow
from src.config import get_pipeline_config
from src.ocr.types import OcrLine
from src.pipelines.calibration import FIELDS, Calibrator, feature_names, load_calibrator
from src.pipelines.receipt_flow import extract_run

TODAY = date(2026, 10, 10)


def constant_calibrator():
    models = {}
    for name in FIELDS:
        n = len(feature_names(name))
        models[name] = {"mean": [0] * n, "std": [1] * n, "coef": [0] * n, "intercept": 0.0}
    return Calibrator(models, {})


def make_run(path):
    path.mkdir()
    pages, gt = [], []
    for i in range(12):
        good = i % 2 == 0
        texts = ["TEST SHOP" if good else "OTHER SHOP", "Ngày: 14/08/2020", "Tổng cộng 50,000" if good else "Tổng cộng 60,000"]
        lines = [OcrLine([[0, y], [200, y], [200, y + 24], [0, y + 24]], text=text, rec_score=0.95 if good else 0.7)
                 for y, text in zip((0, 80, 160), texts, strict=True)]
        image_id = f"{i}.jpg"
        pages.append({"image_id": image_id, "lines": [ln.to_dict() for ln in lines]})
        gt.append({"image_id": image_id, "group_id": f"g{i // 2}", "regions": [
            {"label": label, "role": "value", "text": text, "bbox": [0, y, 200, 24]}
            for label, text, y in (("SELLER", "TEST SHOP", 0), ("TIMESTAMP", "14/08/2020", 80), ("TOTAL_COST", "50,000", 160))]})
    (path / "dataset.json").write_text(json.dumps({"split": "val", "pages": gt}), encoding="utf-8")
    (path / "pages_vietocr.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pages), encoding="utf-8")
    (path / "extraction_vietocr.jsonl").write_text("bản đã hiệu chỉnh cần giữ nguyên\n", encoding="utf-8")
    return path


def test_a8_calibration_cli_preserves_run_and_uses_config_tau_low(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path / "val-none")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    cfg = get_pipeline_config().model_copy(deep=True)
    cfg.routing.tau_low, cfg.routing.tau_high = 0.73, 0.9
    monkeypatch.setattr(calibrate_confidence, "get_pipeline_config", lambda: cfg)
    out = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", ["calibrate_confidence.py", "--runs", str(run), "--out", str(out), "--folds", "2", "--repeats", "1"])
    calibrate_confidence.main()
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert "tau_low = 0.73" in capsys.readouterr().out
    cal = load_calibrator(out)
    assert cal.meta["tau_low"] == 0.73 and cal.is_in_sample(run)


def test_a8_extract_cli_warns_for_in_sample_run_and_can_write_elsewhere(tmp_path, monkeypatch):
    run = make_run(tmp_path / "val-none")
    # Chỉ cần mô hình hằng và metadata tên run cũ để tái hiện cảnh báo, không phụ thuộc weights trên máy.
    cal = constant_calibrator()
    cal.meta = {"runs": [run.name]}
    monkeypatch.setattr("src.pipelines.receipt_flow.load_calibrator", lambda: cal)
    out = tmp_path / "report"
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(sys, "argv", ["run_receipt_flow.py", "extract", "--work", str(run), "--out", str(out), "--n-resamples", "10"])
    with pytest.warns(UserWarning, match="in-sample"):
        run_receipt_flow.main()
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert (out / "flow_report_vietocr.json").exists()


def test_a8_extract_in_memory_does_not_replace_calibrated_results(tmp_path):
    run = make_run(tmp_path / "source")
    before = (run / "extraction_vietocr.jsonl").read_bytes()
    results = extract_run(run, today=TODAY, calibrate=False, write=False)
    assert len(results) == 12 and all(r["field_probs"] is None for r in results)
    assert (run / "extraction_vietocr.jsonl").read_bytes() == before


def test_a8_in_sample_paths_do_not_confuse_identically_named_runs(tmp_path):
    cal = constant_calibrator()
    source = tmp_path / "a" / "val-none"
    cal.meta = {"runs": ["val-none"], "run_paths": [str(source.resolve())]}
    assert cal.is_in_sample(source)
    assert not cal.is_in_sample(tmp_path / "b" / "val-none")
