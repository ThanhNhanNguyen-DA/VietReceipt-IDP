"""Staged OCR baseline CLI; see docs/OCR_BASELINE.md."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr.baseline import prepare, recognize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare", help="Generate shared E1/E2 PNG crops in a new directory")
    p.add_argument("--config", type=Path, default=ROOT / "configs/ocr.yaml")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/processed/mcocr_clean/all.jsonl")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test", "test_seen"], default="val")
    p.add_argument("--preproc", choices=["none", "rules", "full"], default="none")
    p.add_argument("--mode", choices=["e1", "e2", "both"], default="both")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--device", default="gpu:0", help="Paddle device: gpu:0 or cpu")
    p = subs.add_parser("recognize", help="Run one recognizer on the shared crops")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--engine", choices=["paddle", "vietocr"], required=True)
    p.add_argument("--device", help="Paddle: gpu:0/cpu; VietOCR: cuda:0/cpu")
    p.add_argument("--batch-order", choices=["width", "input"], default="width", help="Group similar crop widths; restore sample order in output")
    p = subs.add_parser("evaluate", help="Score annotated regions, write JSON/CSV and optional MLflow run")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--engines", nargs="+", choices=["paddle", "vietocr"], default=["paddle", "vietocr"])
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-mlflow", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "prepare":
            args["config_path"] = args.pop("config")
            prepare(**args)
        elif command == "recognize":
            recognize(**args)
        else:
            from src.ocr.report import evaluate

            report = evaluate(**args)
            for engine, modes in report["engines"].items():
                for mode in ("e1", "e2"):
                    if mode in modes:
                        m = modes[mode]["overall"]
                        print(f"{engine} {mode}: CER={m['cer_exact']['value']:.2%} WER={m['wer_exact']['value']:.2%} coverage={m['region_coverage']:.2%}")
    except (ValueError, OSError) as e:
        parser.exit(1, f"{e}\n")


if __name__ == "__main__":
    main()
