"""E1 recognition and E2 det+rec evaluation on annotated MC-OCR regions only."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import paired_ratio_diff_ci, ratio_ci
from src.ocr.baseline import file_hash, read_jsonl, write_json
from src.ocr.geometry import assign_lines_to_regions, reading_order
from src.ocr.metrics import VARIANTS, score_pair
from src.ocr.types import OcrLine
from src.validation.normalize import parse_date, parse_time, parse_vnd


def evaluation_rows(bundle: dict, predictions: list[dict]) -> list[dict]:
    ids = [p["sample_id"] for p in predictions]
    expected = {s["sample_id"] for s in bundle["samples"]}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Predictions must contain exactly one result per shared crop (no missing/extra/duplicate IDs)")
    by_id = {p["sample_id"]: p for p in predictions}
    e1_ids = {(s["image_id"], s["region_id"]): s["sample_id"] for s in bundle["samples"] if s["mode"] == "e1"}
    rows = []
    for page in bundle["pages"]:
        lines = [OcrLine(polygon=ln["polygon"], det_score=ln["det_score"], text=by_id[ln["sample_id"]]["text"],
                         rec_score=by_id[ln["sample_id"]]["rec_score"]) for ln in page["lines"]]
        assigned = assign_lines_to_regions(lines, page["regions"], bundle["config"]["evaluation"]["region_match"]["min_overlap"])
        for i, reg in enumerate(page["regions"]):
            modes = ("e1", "e2") if bundle["mode"] == "both" else (bundle["mode"],)
            for mode in modes:
                if mode == "e1":
                    sample_id = e1_ids[(page["image_id"], reg["region_id"])]
                    hyp = by_id[sample_id]["text"]
                    matched = 1
                else:
                    transform = page.get("meta", {}).get("H")
                    hyp = " ".join(ln.text for ln in reading_order(assigned[i], H=np.asarray(transform) if transform is not None else None))
                    matched = int(bool(assigned[i]))
                rows.append({"image_id": page["image_id"], "group_id": page["group_id"], "quality": page["quality"],
                             "region_id": reg["region_id"], "mode": mode, "label": reg["label"], "role": reg.get("role"),
                             "ref": reg["text"], "hyp": hyp, "matched": matched, **score_pair(reg["text"], hyp)})
    return rows


def normalized_fields(rows):
    result = {}
    for label, name, parser in (("TOTAL_COST", "total_vnd", parse_vnd), ("TIMESTAMP", "date", parse_date), ("TIMESTAMP", "time", parse_time)):
        pairs = [(parser(r["ref"]), parser(r["hyp"])) for r in rows if r["label"] == label and r["role"] == "value"]
        pairs = [(a, b) for a, b in pairs if a is not None]
        result[name] = {"valid_reference_regions": len(pairs), "matches": sum(a == b for a, b in pairs),
                        "accuracy": sum(a == b for a, b in pairs) / len(pairs) if pairs else None}
    return result


def aggregate(rows: list[dict], n_resamples: int, confidence: float, seed: int) -> dict:
    if not rows:
        raise ValueError("No valid annotated regions to evaluate")
    out = {"regions": len(rows), "documents": len({r['image_id'] for r in rows}),
           "region_coverage": sum(r["matched"] for r in rows) / len(rows),
           "exact_match_rate": sum(r["exact_match"] for r in rows) / len(rows)}
    for variant in VARIANTS:
        for unit, metric in (("char", "cer"), ("word", "wer")):
            out[f"{metric}_{variant}"] = ratio_ci([r["group_id"] for r in rows], [r[f"{unit}_err_{variant}"] for r in rows],
                                                  [r[f"{unit}_len_{variant}"] for r in rows], n_resamples, confidence, seed)
    return out


def summarize(rows: list[dict], n_resamples=1000, confidence=0.95, seed=42) -> dict:
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    report = {}
    for mode, subset in by_mode.items():
        by_label = defaultdict(list)
        by_role, by_quality = defaultdict(list), defaultdict(list)
        for row in subset:
            by_label[row["label"]].append(row)
            by_role[row["role"] or "unknown"].append(row)
            by_quality["low" if row["quality"] < 0.5 else "normal"].append(row)
        report[mode] = {"overall": aggregate(subset, n_resamples, confidence, seed),
                        "by_label": {label: aggregate(rs, n_resamples, confidence, seed) for label, rs in by_label.items()},
                        "by_role": {role: aggregate(rs, n_resamples, confidence, seed) for role, rs in by_role.items()},
                        "by_quality": {quality: aggregate(rs, n_resamples, confidence, seed) for quality, rs in by_quality.items()}}
    return report


def evaluate(work: Path, engines: list[str], n_resamples=1000, confidence=0.95, seed=42, log_mlflow=False) -> dict:
    if not engines or len(engines) != len(set(engines)):
        raise ValueError("Choose one or two distinct engines")
    bundle = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
    all_rows, runs, reports = {}, {}, {}
    for engine in engines:
        runs[engine] = json.loads((work / f"run_{engine}.json").read_text(encoding="utf-8"))
        if runs[engine]["dataset_sha256"] != file_hash(work / "dataset.json"):
            raise ValueError(f"Results belong to a different dataset: {engine}")
        if runs[engine]["predictions_sha256"] != file_hash(work / f"predictions_{engine}.jsonl"):
            raise ValueError(f"Predictions changed after recognition: {engine}")
        predictions = read_jsonl(work / f"predictions_{engine}.jsonl")
        rows = evaluation_rows(bundle, predictions)
        all_rows[engine] = rows
        reports[engine] = summarize(rows, n_resamples, confidence, seed)
        # Estimate compute time for E2 from shared preparation plus batched recognition.
        # Includes every detected line, even those outside annotated regions.
        e2_images = defaultdict(float)
        by_id = {p["sample_id"]: p for p in predictions}
        for sample in bundle["samples"]:
            if sample["mode"] == "e2":
                e2_images[sample["image_id"]] += by_id[sample["sample_id"]]["timing_s"]
        if bundle["mode"] != "e1":
            times = [p["timing_s"]["preprocessing"] + p["timing_s"]["detection"] + e2_images[p["image_id"]] for p in bundle["pages"]]
            reports[engine]["e2_compute_s"] = {"mean": float(np.mean(times)), "p95": float(np.quantile(times, 0.95)),
                                             "note": "Estimate using amortized batch recognition; orientation probing excluded; not API/end-to-end wall latency."}
    paired = {}
    if len(engines) == 2:
        a, b = engines
        for mode in reports[a].keys() & reports[b].keys() & {"e1", "e2"}:
            ra = [r for r in all_rows[a] if r["mode"] == mode]
            rb = [r for r in all_rows[b] if r["mode"] == mode]
            paired[mode] = {"order": f"{a} minus {b}"}
            for unit, metric in (("char", "cer"), ("word", "wer")):
                paired[mode][metric] = paired_ratio_diff_ci([r["group_id"] for r in ra],
                    [r[f"{unit}_err_exact"] for r in ra], [r[f"{unit}_len_exact"] for r in ra],
                    [r[f"{unit}_err_exact"] for r in rb], [r[f"{unit}_len_exact"] for r in rb], n_resamples, confidence, seed)
    report = {"split": bundle["split"], "preproc": bundle["preproc"], "manifest_sha256": bundle["manifest_sha256"],
              "config": bundle["config"], "bootstrap": {"n_resamples": n_resamples, "confidence": confidence, "seed": seed},
              "scope": "MC-OCR annotated regions only. Unannotated text is excluded; no full-page detection precision/recall claim.",
              "skipped_regions": sum(p["skipped_regions"] for p in bundle["pages"]), "engines": reports, "paired": paired, "runs": runs}
    if "orientation" in bundle:
        report["orientation"] = {k: v for k, v in bundle["orientation"].items() if k != "pages"}
        report["orientation"]["sha256"] = bundle["orientation_sha256"]
        report["orientation"]["rotated_pages"] = [p["image_id"] for p in bundle["pages"]
                                                  if p["meta"]["orientation"]["degrees"] == 180]
    write_json(work / "report.json", report)
    for engine, rows in all_rows.items():
        with (work / f"errors_{engine}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if log_mlflow:
        _log_mlflow(work, report)
    return report


def _log_mlflow(work: Path, report: dict):
    import mlflow

    from src.config import get_settings

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("ocr-baseline")
    with mlflow.start_run(run_name=f"{report['split']}-{report['preproc']}"):
        mlflow.log_params({"split": report["split"], "preproc": report["preproc"], "manifest_sha256": report["manifest_sha256"]})
        for engine, modes in report["engines"].items():
            for mode in ("e1", "e2"):
                if mode in modes:
                    metrics = modes[mode]["overall"]
                    mlflow.log_metrics({f"{engine}_{mode}_cer": metrics["cer_exact"]["value"],
                                        f"{engine}_{mode}_wer": metrics["wer_exact"]["value"],
                                        f"{engine}_{mode}_coverage": metrics["region_coverage"]})
        mlflow.log_artifact(str(work / "report.json"))
        for engine in report["engines"]:
            mlflow.log_artifact(str(work / f"errors_{engine}.csv"))
