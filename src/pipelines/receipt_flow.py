"""Luồng đầu tiên trên ảnh MC-OCR: dòng OCR (VietOCR) -> trích trường (luật) -> ReceiptDraft -> kiểm tra -> routing -> chấm điểm theo GT.

Hai bước OCR (detect bằng Paddle, nhận dạng bằng VietOCR) chạy ở môi trường riêng qua `scripts/ocr_baseline.py`; module này chỉ
đọc `pages_<engine>.jsonl` do bước đó ghi ra nên chạy được trong `.venv` chung.
MC-OCR không có nhãn dòng hàng, nên trường bắt buộc ở đây là tên người bán, ngày và tổng tiền (không có `items`).
"""
import datetime as dt
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.config import PipelineConfig, get_pipeline_config
from src.evaluation.bootstrap import ratio_ci
from src.extraction.rules import FieldResult, extract_fields
from src.ocr.baseline import read_jsonl, write_json
from src.ocr.metrics import char_edit
from src.ocr.types import OcrLine
from src.validation.normalize import parse_date, parse_time, parse_vnd
from src.validation.schemas import MerchantDraft, ReceiptDraft, Routing, TransactionDraft

MODEL_VERSION = "rules-v1+paddle-det+vietocr"
REQUIRED_FIELDS = {"merchant.name": "merchant_name", "transaction.date": "date", "total": "total"}
CLOSE_CER = 0.25   # tên/địa chỉ coi là "đúng" khi CER không phân biệt hoa/thường <= 25%


def build_draft(fields: dict[str, FieldResult], doc_confidence: float | None = None, routing: Routing | None = None) -> ReceiptDraft:
    value = lambda k: fields[k].value  # noqa: E731
    return ReceiptDraft(
        merchant=MerchantDraft(name=value("merchant_name"), address=value("merchant_address")),
        transaction=TransactionDraft(date=dt.date.fromisoformat(value("date")) if value("date") else None,
                                     time=dt.time.fromisoformat(value("time")) if value("time") else None),
        total=Decimal(value("total")) if value("total") is not None else None,
        doc_confidence=doc_confidence, routing=routing, model_version=MODEL_VERSION)


def validate(fields: dict[str, FieldResult], today: dt.date) -> list[str]:
    issues = [f"missing:{path}" for path, key in REQUIRED_FIELDS.items() if fields[key].value is None]
    if fields["date"].value and dt.date.fromisoformat(fields["date"].value) > today:
        issues.append("date_in_future")
    if fields["total"].value is not None and fields["total"].value <= 0:
        issues.append("total_not_positive")
    return issues


def doc_confidence(fields: dict[str, FieldResult]) -> float:
    """Min độ tin cậy theo các trường bắt buộc (kế hoạch mục 5); thiếu trường -> 0."""
    return min(fields[key].confidence for key in REQUIRED_FIELDS.values())


def route(conf: float, issues: list[str], cfg: PipelineConfig) -> Routing:
    if issues or conf < cfg.routing.tau_low:
        return Routing.human_review
    return Routing.auto_accept if conf >= cfg.routing.tau_high else Routing.accept_with_warning


def process_page(page: dict, cfg: PipelineConfig, today: dt.date) -> dict[str, Any]:
    lines = [OcrLine(**ln) for ln in page["lines"]]
    fields = extract_fields(lines, today)
    issues = validate(fields, today)
    conf = doc_confidence(fields)
    routing = route(conf, issues, cfg)
    scores = [ln.rec_score for ln in lines if ln.rec_score is not None]
    return {
        "image_id": page["image_id"],
        "draft": build_draft(fields, conf, routing).model_dump(mode="json"),
        "fields": {k: {"value": f.value, "confidence": round(f.confidence, 4), "rule": f.rule, "evidence": f.evidence} for k, f in fields.items()},
        "doc_confidence": round(conf, 4), "routing": routing.value, "validation_issues": issues,
        # ocr_conf lưu từ M2 để tune ngưỡng ở M5
        "ocr": {"lines": len(lines), "mean_rec_score": round(sum(scores) / len(scores), 4) if scores else None},
    }


def extract_run(work: Path, engine: str = "vietocr", today: dt.date | None = None, cfg: PipelineConfig | None = None) -> list[dict]:
    cfg, today = cfg or get_pipeline_config(), today or dt.date.today()
    results = [process_page(p, cfg, today) for p in read_jsonl(work / f"pages_{engine}.jsonl")]
    (work / f"extraction_{engine}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    return results


# ---------- Chấm điểm theo GT ----------
def _gt_values(page: dict, label: str) -> list[str]:
    return [r["text"] for r in page["regions"] if r["label"] == label and r.get("role") == "value" and r["text"].strip()]


def _best_edit(page: dict, label: str, hyp: str | None):
    """Tên người bán: MC-OCR hay gán nhiều vùng cho cách viết khác nhau của cùng một tên, nên so với vùng gần nhất."""
    edits = [char_edit(t, hyp or "", "nocase") for t in _gt_values(page, label)]
    return min(edits, key=lambda e: e.errors / max(e.ref_len, 1)) if edits else None


def _gt_joined(page: dict, label: str) -> str:
    """Các vùng GT của một nhãn nối theo thứ tự đọc (địa chỉ/tên có thể bị chia nhiều vùng)."""
    regions = sorted((r for r in page["regions"] if r["label"] == label and r.get("role") == "value" and r["text"].strip()),
                     key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return " ".join(r["text"].strip() for r in regions)


def score_page(page: dict, result: dict) -> dict[str, Any]:
    """Một dòng điểm cho mỗi ảnh. Trường không có tham chiếu hợp lệ -> None (loại khỏi mẫu số)."""
    v = {k: result["fields"][k]["value"] for k in result["fields"]}
    row: dict[str, Any] = {"image_id": page["image_id"], "group_id": page["group_id"], "routing": result["routing"],
                           "doc_confidence": result["doc_confidence"]}
    for name, label, parse in (("total", "TOTAL_COST", parse_vnd), ("date", "TIMESTAMP", parse_date), ("time", "TIMESTAMP", parse_time)):
        gt = {parse(t) for t in _gt_values(page, label)} - {None}
        row[name] = None if not gt else int(v[name] in gt)
    for name, label, key in (("seller", "SELLER", "merchant_name"), ("address", "ADDRESS", "merchant_address")):
        if name == "seller":
            edit = _best_edit(page, label, v[key])
        else:  # địa chỉ trải nhiều dòng: so với các vùng GT nối theo thứ tự đọc
            ref = _gt_joined(page, label)
            edit = char_edit(ref, v[key] or "", "nocase") if ref else None
        row[f"{name}_err"], row[f"{name}_len"] = (edit.errors, edit.ref_len) if edit else (None, None)
        row[name] = None if edit is None else int(edit.errors <= CLOSE_CER * edit.ref_len)
    parts = [row[k] for k in ("total", "date", "seller") if row[k] is not None]
    row["doc_correct"] = int(all(parts)) if parts else None
    return row


def _acc(rows: list[dict], key: str, n_resamples: int, seed: int) -> dict[str, Any]:
    r = [x for x in rows if x[key] is not None]
    if not r:
        return {"n": 0}
    ci = ratio_ci([x["group_id"] for x in r], [x[key] for x in r], [1] * len(r), n_resamples, 0.95, seed)
    return {"n": len(r), "correct": sum(x[key] for x in r), "accuracy": ci["value"], "ci95": [ci["low"], ci["high"]]}


def evaluate(work: Path, engine: str = "vietocr", n_resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    pages = {p["image_id"]: p for p in json.loads((work / "dataset.json").read_text(encoding="utf-8"))["pages"]}
    results = read_jsonl(work / f"extraction_{engine}.jsonl")
    rows = [score_page(pages[r["image_id"]], r) for r in results]
    report: dict[str, Any] = {"engine": engine, "documents": len(rows), "model_version": MODEL_VERSION,
                              "fields": {k: _acc(rows, k, n_resamples, seed) for k in ("total", "date", "time", "seller", "address", "doc_correct")}}
    for name in ("seller", "address"):
        r = [x for x in rows if x[f"{name}_len"]]
        if r:
            ci = ratio_ci([x["group_id"] for x in r], [x[f"{name}_err"] for x in r], [x[f"{name}_len"] for x in r], n_resamples, 0.95, seed)
            report["fields"][name]["cer_nocase"] = {"value": ci["value"], "ci95": [ci["low"], ci["high"]]}
    by_route: dict[str, list[dict]] = defaultdict(list)
    for x in rows:
        by_route[x["routing"]].append(x)
    report["routing"] = {k: {"documents": len(v), "doc_accuracy": (sum(x["doc_correct"] for x in v if x["doc_correct"] is not None)
                                                                   / max(1, sum(x["doc_correct"] is not None for x in v)))}
                         for k, v in by_route.items()}
    scored = sorted((x for x in rows if x["doc_correct"] is not None), key=lambda x: -x["doc_confidence"])
    report["risk_coverage"] = [{"tau": t, "coverage": sum(x["doc_confidence"] >= t for x in scored) / len(scored),
                                "error_rate": (lambda sel: 1 - sum(x["doc_correct"] for x in sel) / len(sel) if sel else None)(
                                    [x for x in scored if x["doc_confidence"] >= t])}
                               for t in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)]
    write_json(work / f"flow_report_{engine}.json", report)
    (work / f"flow_rows_{engine}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return report
