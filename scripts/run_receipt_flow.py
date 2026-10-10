"""Luồng đầu tiên trên ảnh MC-OCR: tiền xử lý -> Paddle detect -> VietOCR -> trích trường (luật) -> routing -> chấm điểm.

  run      chạy cả luồng (detect ở .venv-ocr, VietOCR ở .venv-kie, trích trường ở môi trường hiện tại) vào thư mục mới
  extract  chỉ trích trường + chấm điểm trên thư mục OCR đã có (pages_vietocr.jsonl + dataset.json)
Chi tiết: docs/RECEIPT_FLOW.md.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.receipt_flow import evaluate, extract_run  # noqa: E402


def sh(venv: str, *args: str):
    cmd = [str(ROOT / venv / "bin/python"), str(ROOT / "scripts/ocr_baseline.py"), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def summary(report: dict):
    print(f"\n{report['documents']} ảnh, {report['model_version']}")
    for name, f in report["fields"].items():
        if not f.get("n"):
            continue
        cer = f.get("cer_nocase")
        extra = f"  CER(nocase) {100 * cer['value']:.1f}%" if cer else ""
        print(f"  {name:<10} {f['correct']:>4}/{f['n']:<4} {100 * f['accuracy']:5.1f}%  CI95 [{100 * f['ci95'][0]:.1f}, {100 * f['ci95'][1]:.1f}]{extra}")
    print("\nRouting (doc đúng = total, date, seller cùng đúng):")
    for k, v in sorted(report["routing"].items()):
        print(f"  {k:<20} {v['documents']:>4} ảnh, đúng {100 * v['doc_accuracy']:.1f}%")
    print("\nRisk-coverage theo doc_confidence:")
    for r in report["risk_coverage"]:
        err = "-" if r["error_rate"] is None else f"{100 * r['error_rate']:.1f}%"
        print(f"  tau >= {r['tau']:.2f}  coverage {100 * r['coverage']:5.1f}%  lỗi {err}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = p.add_subparsers(dest="cmd", required=True)
    r = subs.add_parser("run", help="Chạy cả luồng vào thư mục mới")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", choices=["train", "val"], default="val", help="test bị khoá đến khi chốt cấu hình")
    r.add_argument("--limit", type=int, default=0)
    e = subs.add_parser("extract", help="Trích trường + chấm điểm trên run OCR có sẵn")
    e.add_argument("--work", type=Path, required=True)
    for s in (r, e):
        s.add_argument("--n-resamples", type=int, default=1000)
    args = p.parse_args()
    work = args.out if args.cmd == "run" else args.work
    if args.cmd == "run":
        limit = ["--limit", str(args.limit)] if args.limit else []
        sh(".venv-ocr", "prepare", "--out", str(work), "--split", args.split, "--mode", "e2", *limit)
        sh(".venv-kie", "recognize", "--work", str(work), "--engine", "vietocr")
    extract_run(work)
    summary(evaluate(work, n_resamples=args.n_resamples))
    print(f"\nKết quả từng ảnh: {work}/extraction_vietocr.jsonl; báo cáo: {work}/flow_report_vietocr.json")


if __name__ == "__main__":
    main()
