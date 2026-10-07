"""Summarize completed validation runs without loading any OCR model."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap import paired_ratio_diff_ci  # noqa: E402
from src.ocr.baseline import write_json  # noqa: E402
from src.ocr.report import normalized_fields  # noqa: E402


def percent(x):
    return f"{100 * x:.2f}%"


def plot_cer(reports, out: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = list(reports)
    positions = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, mode, title in zip(axes, ("e1", "e2"), ("E1: GT crop recognition", "E2: detection + recognition"), strict=True):
        for offset, engine, color in ((-0.18, "paddle", "#2463a5"), (0.18, "vietocr", "#d77722")):
            metrics = [reports[v]["engines"][engine][mode]["overall"]["cer_exact"] for v in variants]
            x = positions + offset
            ax.bar(x, [100*m["value"] for m in metrics], width=0.34, color=color, label=engine)
            for xx, metric in zip(x, metrics, strict=True):
                low, high = 100 * metric["low"], 100 * metric["high"]
                ax.vlines(xx, low, high, color="#222", linewidth=1)
                ax.hlines([low, high], xx - 0.035, xx + 0.035, color="#222", linewidth=1)
            for xx, metric in zip(x, metrics, strict=True):
                ax.text(xx, 100*metric["value"] + 0.5, f"{100*metric['value']:.1f}", ha="center", fontsize=8)
        ax.axhline(5, color="#555", linestyle="--", linewidth=1, label="CER target 5%")
        ax.set_xticks(positions, variants)
        ax.set_title(title)
        ax.set_xlabel("Preprocessing")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Corpus CER (%) with merchant bootstrap 95% CI")
    axes[1].legend(fontsize=8)
    fig.suptitle("MC-OCR validation: shared crops, exact NFC text")
    fig.savefig(out / "cer_comparison.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--works", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/ocr/validation-summary")
    p.add_argument("--plot", action="store_true", help="Export a standalone CER comparison figure (requires matplotlib)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports, datasets, rows = {}, {}, {}
    for work in args.works:
        report = json.loads((work / "report.json").read_text(encoding="utf-8"))
        dataset = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
        if report["split"] != "val" or dataset["mode"] != "both" or set(report["engines"]) != {"paddle", "vietocr"}:
            raise ValueError("Comparison requires completed A/B E1+E2 validation runs")
        name = report["preproc"]
        if name in reports:
            raise ValueError(f"Duplicate preprocessing variant: {name}")
        reports[name], datasets[name] = report, dataset
        for engine in report["engines"]:
            with (work / f"errors_{engine}.csv").open(encoding="utf-8", newline="") as f:
                rows[name, engine] = list(csv.DictReader(f))
    first = next(iter(datasets.values()))
    expected = {(pg["image_id"], reg["region_id"], reg["text"]) for pg in first["pages"] for reg in pg["regions"]}
    for name, dataset in datasets.items():
        current = {(pg["image_id"], reg["region_id"], reg["text"]) for pg in dataset["pages"] for reg in pg["regions"]}
        if current != expected or dataset["manifest_sha256"] != first["manifest_sha256"]:
            raise ValueError(f"Reference populations differ: {name}")
    analysis = {"pages": len(first["pages"]), "groups": len({pg["group_id"] for pg in first["pages"]}),
                "regions": len(expected), "skipped_regions": reports[next(iter(reports))]["skipped_regions"],
                "normalized_fields": {}, "preprocessing_paired": {}}
    for (variant, engine), samples in rows.items():
        analysis["normalized_fields"][f"{variant}/{engine}"] = {
            mode: normalized_fields([r for r in samples if r["mode"] == mode]) for mode in ("e1", "e2")}
    if "none" in reports:
        for variant in reports.keys() - {"none"}:
            for engine in ("paddle", "vietocr"):
                for mode in ("e1", "e2"):
                    base = [r for r in rows["none", engine] if r["mode"] == mode]
                    candidate = {(r["image_id"], r["region_id"]): r for r in rows[variant, engine] if r["mode"] == mode}
                    paired = [candidate[r["image_id"], r["region_id"]] for r in base]
                    analysis["preprocessing_paired"][f"{variant}/{engine}/{mode}"] = paired_ratio_diff_ci(
                        [r["group_id"] for r in base], [int(r["char_err_exact"]) for r in paired], [int(r["char_len_exact"]) for r in paired],
                        [int(r["char_err_exact"]) for r in base], [int(r["char_len_exact"]) for r in base], seed=42)
    lines = ["# OCR validation benchmark", "", f"{analysis['pages']} ảnh, {analysis['groups']} nhóm merchant/template, "
             f"{analysis['regions']} vùng GT hợp lệ; loại {analysis['skipped_regions']} vùng. Chỉ chạy validation; chưa mở test.", "",
             "A = Paddle det + Paddle rec; B = cùng Paddle det/crop + VietOCR rec. Trong mỗi variant A/B dùng cùng file PNG.", "",
             "E1: recognition trên crop GT. E2: det+rec ghép về vùng có nhãn. Không có ground truth transcript toàn trang.", "",
             "## Kết quả tổng hợp", "", "| Preproc | Recognizer | E1 CER | E1 WER | E2 CER | E2 WER | E2 coverage |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for variant, report in reports.items():
        for engine, modes in report["engines"].items():
            a, b = modes["e1"]["overall"], modes["e2"]["overall"]
            lines.append(f"| {variant} | {engine} | {percent(a['cer_exact']['value'])} | {percent(a['wer_exact']['value'])} | "
                         f"{percent(b['cer_exact']['value'])} | {percent(b['wer_exact']['value'])} | {percent(b['region_coverage'])} |")
    lines += ["", "CER/WER exact giữ hoa/thường và dấu sau NFC + gom whitespace. Target kế hoạch: CER ≤ 5%, WER ≤ 12%.", "",
              "## CER và CI 95%", "", "| Preproc | Recognizer | Mode | CER [95% CI] | Value-only CER | Nocase CER | Nodiac CER |",
              "|---|---|---|---|---:|---:|---:|"]
    for variant, report in reports.items():
        for engine, modes in report["engines"].items():
            for mode in ("e1", "e2"):
                v = modes[mode]["overall"]; c = v["cer_exact"]
                lines.append(f"| {variant} | {engine} | {mode} | {percent(c['value'])} [{percent(c['low'])}, {percent(c['high'])}] | "
                             f"{percent(modes[mode]['by_role']['value']['cer_exact']['value'])} | {percent(v['cer_nocase']['value'])} | {percent(v['cer_nodiac']['value'])} |")
    lines += ["", "## So sánh recognizer trên cùng crop", "", "| Preproc | Mode | ΔCER Paddle−VietOCR [95% CI], điểm % |",
              "|---|---|---|"]
    for variant, report in reports.items():
        for mode in ("e1", "e2"):
            v = report["paired"][mode]["cer"]
            lines.append(f"| {variant} | {mode} | {100*v['diff']:.2f} [{100*v['low']:.2f}, {100*v['high']:.2f}] |")
    lines += ["", "Δ âm: Paddle có CER thấp hơn. CI được bootstrap theo nhóm, 1.000 lần, seed 42.", "",
              "## Giá trị sau chuẩn hóa", "", "| Preproc/Recognizer | Mode | TOTAL VND | DATE | TIME |", "|---|---|---|---|---|"]
    for config, modes in analysis["normalized_fields"].items():
        for mode, metrics in modes.items():
            cells = [f"{m['matches']}/{m['valid_reference_regions']} ({percent(m['accuracy'])})" if m["accuracy"] is not None else "N/A" for m in metrics.values()]
            lines.append(f"| {config} | {mode} | {' | '.join(cells)} |")
    lines += ["", "Chỉ tính các vùng value có tham chiếu parse được; đây là metric từng vùng, chưa phải document exact match.", "",
              "## Giới hạn và artifacts", "",
              "- Vùng GT không được detect vẫn tính deletion. Dòng ngoài vùng gán nhãn không thể phân loại false positive.",
              "- Coverage là vùng có ít nhất một dòng khớp theo hình học; không phải detection recall toàn trang.",
              "- Khi detector gộp key và value vào một dòng, hypothesis có thể chứa chữ ngoài vùng GT; xem CSV và overlay trước khi quy mọi lỗi E2 cho recognizer.",
              "- Role key/value đến từ heuristic của manifest, chưa phải nhãn role đã được người kiểm tra. Các tỷ lệ sau chuẩn hóa dùng mọi tham chiếu parse được, có thể bao gồm lỗi annotation.",
              "- Batch order width được cố định cho cả hai engine; thay đổi cách chia batch có thể ảnh hưởng padding và output của recognizer.",
              "- Thời gian E2 trong report là ước lượng compute, loại I/O/khởi tạo model và amortized recognition; chưa phải latency API.",
              "- Mỗi thư mục run có dataset/config/versions/hash, predictions, pages, report JSON và errors CSV.", ""]
    write_json(args.out / "comparison.json", {"analysis": analysis, "reports": reports})
    if args.plot:
        plot_cer(reports, args.out)
        lines += ["![CER comparison](cer_comparison.png)", ""]
    (args.out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out / 'comparison.md'}")


if __name__ == "__main__":
    main()
