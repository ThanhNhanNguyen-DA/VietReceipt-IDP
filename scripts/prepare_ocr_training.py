"""Audit and version train/validation OCR crops for both recognizers."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ocr.training_data import finalize_data, prepare_audit, verify_training_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('audit')
    p.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/mcocr_clean/all.jsonl')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--recognition-dir', type=Path, default=ROOT / 'data/processed/mcocr_clean')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/ocr.yaml')
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('finalize')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--corrections', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('verify')
    p.add_argument('--root', type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    try:
        result = {'audit': prepare_audit, 'finalize': finalize_data, 'verify': verify_training_data}[command](**args)
        print(result.get('counts', f"Prepared {len(result.get('samples', []))} audited crops"))
    except (ValueError, OSError) as e:
        parser.exit(1, f'{e}\n')


if __name__ == '__main__':
    main()
