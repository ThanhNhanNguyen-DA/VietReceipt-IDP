"""Launch pinned upstream PaddleOCR training with corpus CER checkpoint selection."""
import argparse
import copy
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.ocr.backends import _download_asset, package_versions  # noqa: E402
from src.ocr.baseline import file_hash, write_json  # noqa: E402
from src.ocr.training import PaddleCorpusMetric  # noqa: E402
from src.ocr.training_data import verify_training_data  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/ocr_finetune.yaml')
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/ocr_review_v1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int)
    args = p.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    metadata = verify_training_data(data)
    settings = yaml.safe_load(args.config.read_text()); params = settings['paddle']
    source = Path(params['source']).resolve()
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != params['source_commit']:
        raise ValueError('PaddleOCR source commit differs from pinned configuration')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Training output must be new/empty')
    weights = _download_asset(params['pretrained_url'], Path(params['pretrained']).resolve())
    template = source / 'configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml'
    cfg = copy.deepcopy(yaml.safe_load(template.read_text()))
    cfg['Global'].update(epoch_num=args.epochs or settings['epochs'], save_model_dir=str(out), save_epoch_step=1,
                         pretrained_model=str(weights), character_dict_path=str(source / 'ppocr/utils/dict/ppocrv5_latin_dict.txt'),
                         distributed=False, use_gpu=True, seed=settings['seed'], print_batch_step=50,
                         eval_batch_step=[0, math.ceil(metadata['counts']['train'] / params['batch_size'])], cal_metric_during_train=False)
    cfg['Optimizer']['lr'].update(learning_rate=params['learning_rate'], warmup_epoch=0)
    cfg['Metric'].update(main_indicator='one_minus_cer', ignore_space=False)
    cfg['Architecture']['Head']['head_list'][1]['NRTRHead']['max_text_length'] = params['max_text_length']
    for sp, name in [('train', 'Train'), ('val', 'Eval')]:
        cfg[name]['dataset'] = {'name': 'SimpleDataSet', 'data_dir': str(data), 'label_file_list': [str(data / f'{sp}.tsv')],
                                'transforms': [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                                               {'MultiLabelEncode': {'max_text_length': params['max_text_length'], 'gtc_encode': 'NRTRLabelEncode'}},
                                               {'RecResizeImg': {'image_shape': params['image_shape'], 'eval_mode': sp == 'val'}},
                                               {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}}]}
        cfg[name].pop('sampler', None)
        cfg[name]['loader'].update(batch_size_per_card=params['batch_size'] if sp == 'train' else 1,
                                    num_workers=0, drop_last=False, shuffle=sp == 'train')
    charset = set((source / 'ppocr/utils/dict/ppocrv5_latin_dict.txt').read_text().splitlines()) | {' '}
    from src.ocr.baseline import read_jsonl

    for sp in ('train', 'val'):
        for sample in read_jsonl(data / f'{sp}.jsonl'):
            if set(sample['text']) - charset or len(sample['text']) > params['max_text_length']:
                raise ValueError(f"Paddle would discard label characters/length: {sample['sample_id']}")
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'training.yml'; config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    write_json(out / 'run.json', {'engine': 'paddle', 'source_commit': sha, 'params': params, 'data': metadata,
                                 'data_metadata_sha256': file_hash(data / 'metadata.json'), 'versions': package_versions(),
                                 'pretrained_sha256': file_hash(weights), 'selection': 'maximum 1 - full validation corpus NFC CER'})
    sys.path.insert(0, str(source))
    # Use upstream architectures/losses/train loop. Only replace its checkpoint metric.
    import ppocr.metrics

    ppocr.metrics.RecMetric = PaddleCorpusMetric
    os.environ.setdefault('FLAGS_fraction_of_gpu_memory_to_use', '0.7')
    sys.argv = [str(source / 'tools/train.py'), '-c', str(config_path)]
    runpy.run_path(str(source / 'tools/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
