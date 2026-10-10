import json
from datetime import date

import pytest

from src.config import get_pipeline_config
from src.extraction.rules import extract_fields
from src.ocr.types import OcrLine
from src.pipelines.receipt_flow import doc_confidence, evaluate, extract_run, process_page, route, score_page, validate
from src.validation.schemas import Routing

TODAY = date(2026, 10, 10)
CFG = get_pipeline_config()


def line(text, x, y, score=0.93):
    return OcrLine(polygon=[[x, y], [x + 200, y], [x + 200, y + 24], [x, y + 24]], det_score=0.9, text=text, rec_score=score)


def page_dict(lines):
    return {"image_id": "a.jpg", "lines": [ln.to_dict() for ln in lines]}


LINES = [line("SIEU THI BACH HOA TONG HOP", 230, 120), line("Số 5 Cẩm Tây - Cẩm Phả - QN", 280, 160), line("Ngày: 14-08-2020 21:31", 200, 320),
         line("Tổng cộng:", 187, 920), line("72,000", 626, 912)]
GT_REGIONS = [{"label": "SELLER", "role": "value", "text": "SIEU THI BACH HOA TONG HOP", "bbox": [230, 120, 200, 24]},
              {"label": "ADDRESS", "role": "value", "text": "Số 5 Cẩm Tây - Cẩm Phả - QN", "bbox": [280, 160, 200, 24]},
              {"label": "TIMESTAMP", "role": "value", "text": "Ngày: 14-08-2020 21:31", "bbox": [200, 320, 200, 24]},
              {"label": "TOTAL_COST", "role": "value", "text": "72.000", "bbox": [626, 912, 200, 24]}]


@pytest.mark.parametrize("conf,issues,expected", [
    (0.95, [], Routing.auto_accept), (0.80, [], Routing.accept_with_warning), (0.50, [], Routing.human_review),
    (0.95, ["missing:total"], Routing.human_review),
])
def test_route(conf, issues, expected):
    assert route(conf, issues, CFG) == expected


def test_validate_and_doc_confidence_use_required_fields_only():
    fields = extract_fields(LINES, TODAY)
    assert validate(fields, TODAY) == []
    assert doc_confidence(fields) == min(fields[k].confidence for k in ("merchant_name", "date", "total"))
    assert validate(extract_fields([], TODAY), TODAY) == ["missing:merchant.name", "missing:transaction.date", "missing:total"]


def test_process_page_builds_draft_and_keeps_ocr_conf():
    out = process_page(page_dict(LINES), CFG, TODAY)
    assert out["draft"]["total"] == "72000" and out["draft"]["transaction"]["date"] == "2020-08-14"
    assert out["draft"]["merchant"]["name"] == "SIEU THI BACH HOA TONG HOP"
    assert out["ocr"]["mean_rec_score"] == 0.93 and out["routing"] in {r.value for r in Routing}
    json.dumps(out)  # tuần tự hoá được


def test_score_page_matches_total_across_number_formats_and_flags_misses():
    page = {"image_id": "a.jpg", "group_id": "g1", "regions": GT_REGIONS}
    row = score_page(page, process_page(page_dict(LINES), CFG, TODAY))
    assert (row["total"], row["date"], row["seller"], row["address"], row["doc_correct"]) == (1, 1, 1, 1, 1)
    assert row["time"] == 1                        # giờ 21:31 nằm cùng dòng với ngày
    no_time = {**page, "regions": [r for r in GT_REGIONS if r["label"] != "TIMESTAMP"]}
    assert score_page(no_time, process_page(page_dict(LINES), CFG, TODAY))["time"] is None   # không có GT -> loại khỏi mẫu số
    wrong = process_page(page_dict([line("Tổng cộng: 99,000", 187, 920)]), CFG, TODAY)
    assert score_page(page, wrong)["total"] == 0 and score_page(page, wrong)["doc_correct"] == 0


def test_extract_run_and_evaluate_roundtrip(tmp_path):
    (tmp_path / "pages_vietocr.jsonl").write_text(json.dumps(page_dict(LINES)) + "\n", encoding="utf-8")
    (tmp_path / "dataset.json").write_text(json.dumps({"pages": [{"image_id": "a.jpg", "group_id": "g1", "regions": GT_REGIONS}]}), encoding="utf-8")
    assert len(extract_run(tmp_path, today=TODAY)) == 1
    report = evaluate(tmp_path, n_resamples=50)
    assert report["documents"] == 1 and report["fields"]["total"]["accuracy"] == 1.0 and report["fields"]["time"]["n"] == 1
    assert (tmp_path / "flow_report_vietocr.json").exists()
