"""Paired comparison of recognizer checkpoints on the unchanged benchmark references (E1/E2, validation only).

Each candidate must be a `fork` of the base run (same crops, boxes and GT); only the recognizer weights differ.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap import paired_ratio_diff_ci  # noqa: E402
from src.ocr.baseline import write_json  # noqa: E402
from src.ocr.report import normalized_fields  # noqa: E402


def load(work: Path, engine: str) -> tuple[dict, dict, list[dict]]:
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    dataset = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
    if report["split"] != "val" or engine not in report["engines"]:
        raise ValueError(f"{work}: need an evaluated validation run containing {engine}")
    with (work / f"errors_{engine}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return report, dataset, rows


def percent(x):
    return f"{100 * x:.2f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="e.g. outputs/ocr/val-none")
    p.add_argument("--candidates", type=Path, nargs="+", required=True)
    p.add_argument("--engine", choices=["paddle", "vietocr"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    base_report, base_data, base_rows = load(args.base, args.engine)
    crops = {(s["sample_id"], s["sha256"]) for s in base_data["samples"]}
    runs = {args.base.name: (base_report, base_rows)}
    for work in args.candidates:
        report, dataset, rows = load(work, args.engine)
        if work.name in runs:
            raise ValueError(f"Duplicate run name: {work.name}")
        if dataset["manifest_sha256"] != base_data["manifest_sha256"] or {(s["sample_id"], s["sha256"]) for s in dataset["samples"]} != crops:
            raise ValueError(f"{work}: crops or manifest differ from the base run")
        if {(r["image_id"], r["region_id"], r["mode"], r["ref"]) for r in rows} != {(r["image_id"], r["region_id"], r["mode"], r["ref"]) for r in base_rows}:
            raise ValueError(f"{work}: reference regions differ from the base run")
        runs[work.name] = report, rows
    summary = {"base": args.base.name, "engine": args.engine, "runs": {}, "paired_vs_base": {}}
    for name, (report, rows) in runs.items():
        modes = {m: report["engines"][args.engine][m] for m in ("e1", "e2") if m in report["engines"][args.engine]}
        summary["runs"][name] = {
            "checkpoint": report["config"]["recognition"][args.engine],
            "overall": {m: {k: v["overall"][k] for k in ("cer_exact", "wer_exact", "cer_nocase", "cer_nodiac", "region_coverage")}
                        for m, v in modes.items()},
            "by_label_cer": {m: {label: v["cer_exact"]["value"] for label, v in mv["by_label"].items()} for m, mv in modes.items()},
            "normalized_fields": {m: normalized_fields([r for r in rows if r["mode"] == m]) for m in modes}}
        if name == args.base.name:
            continue
        by_key = {(r["image_id"], r["region_id"], r["mode"]): r for r in rows}
        for mode in modes:
            base = [r for r in base_rows if r["mode"] == mode]
            cand = [by_key[r["image_id"], r["region_id"], mode] for r in base]
            summary["paired_vs_base"][f"{name}/{mode}"] = {
                unit: paired_ratio_diff_ci([r["group_id"] for r in base],
                                           [int(r[f"{unit}_err_exact"]) for r in cand], [int(r[f"{unit}_len_exact"]) for r in cand],
                                           [int(r[f"{unit}_err_exact"]) for r in base], [int(r[f"{unit}_len_exact"]) for r in base], seed=42)
                for unit in ("char", "word")}
    lines = [f"# Recognizer checkpoints on benchmark validation ({args.engine})", "",
             "Cùng box/crop và GT gốc của base run; chỉ khác recognizer. Checkpoint được chọn trên validation nên số liệu có bias chọn mẫu.", "",
             "| Run | E1 CER | E1 WER | E2 CER | E2 WER | TOTAL E2 | DATE E2 | TIME E2 |", "|---|---:|---:|---:|---:|---|---|---|"]
    for name, run in summary["runs"].items():
        o, nf = run["overall"], run["normalized_fields"].get("e2", {})
        fields = [f"{m['matches']}/{m['valid_reference_regions']}" for m in nf.values()] or ["N/A"] * 3
        lines.append(f"| {name} | {percent(o['e1']['cer_exact']['value'])} | {percent(o['e1']['wer_exact']['value'])} | "
                     f"{percent(o['e2']['cer_exact']['value'])} | {percent(o['e2']['wer_exact']['value'])} | {' | '.join(fields)} |")
    lines += ["", "| Candidate/mode | ΔCER vs base [95% CI], điểm % | ΔWER vs base [95% CI], điểm % |", "|---|---|---|"]
    for key, d in summary["paired_vs_base"].items():
        c, w = d["char"], d["word"]
        lines.append(f"| {key} | {100*c['diff']:.2f} [{100*c['low']:.2f}, {100*c['high']:.2f}] | "
                     f"{100*w['diff']:.2f} [{100*w['low']:.2f}, {100*w['high']:.2f}] |")
    lines += ["", "Δ âm: candidate tốt hơn base. Bootstrap theo merchant/template, 1.000 lần, seed 42. Test vẫn khoá.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "comparison.json", summary)
    (args.out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
