"""Compare completed recognition runs on exactly the same reviewed validation crops."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap import paired_ratio_diff_ci, ratio_ci  # noqa: E402
from src.ocr.baseline import file_hash, read_jsonl, write_json  # noqa: E402
from src.ocr.metrics import score_pair  # noqa: E402
from src.ocr.training import corpus_metrics  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--works', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    datasets, reports, rows = {}, {}, {}
    expected = None
    for work in args.works:
        name = work.name
        if name in datasets:
            raise ValueError('Run names must be distinct')
        bundle = json.loads((work / 'dataset.json').read_text())
        if bundle['split'] != 'val' or any(s['split'] != 'val' for s in bundle['samples']):
            raise ValueError('Only validation is permitted')
        reference = {(s['sample_id'], s['text'], s['sha256'], s['group_id']) for s in bundle['samples']}
        if expected is not None and reference != expected:
            raise ValueError('Runs have different reference text, crop pixels or merchant groups')
        expected = reference
        datasets[name] = bundle
        reports[name] = {}
        for engine in ('paddle', 'vietocr'):
            path = work / f'predictions_{engine}.jsonl'
            run = json.loads((work / f'run_{engine}.json').read_text())
            if run['dataset_sha256'] != file_hash(work / 'dataset.json') or run['predictions_sha256'] != file_hash(path):
                raise ValueError('Run integrity mismatch')
            predictions = read_jsonl(path); by_id = {r['sample_id']: r for r in predictions}
            if len(by_id) != len(predictions) or set(by_id) != {s['sample_id'] for s in bundle['samples']}:
                raise ValueError('Missing, extra or duplicate predictions')
            samples = [{**s, 'hyp': by_id[s['sample_id']]['text'], **score_pair(s['text'], by_id[s['sample_id']]['text'])}
                       for s in bundle['samples']]
            rows[name, engine] = samples
            result = corpus_metrics([s['text'] for s in samples], [s['hyp'] for s in samples])
            for unit, metric in (('char', 'cer'), ('word', 'wer')):
                result[metric + '_ci'] = ratio_ci([s['group_id'] for s in samples], [s[unit + '_err_exact'] for s in samples],
                                                   [s[unit + '_len_exact'] for s in samples], seed=42)
            reports[name][engine] = result
    paired = {}
    first = next(iter(datasets))
    for name in list(datasets)[1:]:
        for engine in ('paddle', 'vietocr'):
            base = rows[first, engine]; candidate = {s['sample_id']: s for s in rows[name, engine]}
            rs = [candidate[s['sample_id']] for s in base]
            paired[name + '/' + engine] = paired_ratio_diff_ci([s['group_id'] for s in base],
                [s['char_err_exact'] for s in rs], [s['char_len_exact'] for s in rs],
                [s['char_err_exact'] for s in base], [s['char_len_exact'] for s in base], seed=42)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'comparison.json', {'reports': reports, 'paired_cer_candidate_minus_baseline': paired,
               'scope': 'Reviewed provided validation crops only; references were corrected/excluded before training. Not directly comparable to original GT E1/E2 or held-out test.'})
    lines = ['# Reviewed OCR crop validation', '', '| Run | Engine | CER | WER | Exact match |', '|---|---|---:|---:|---:|']
    for name, engines in reports.items():
        for engine, metrics in engines.items():
            lines.append(f"| {name} | {engine} | {metrics['cer']:.2%} | {metrics['wer']:.2%} | {metrics['exact_match']:.2%} |")
    (args.out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
