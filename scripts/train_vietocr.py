"""Fine-tune the cached VietOCR checkpoint; select by full validation corpus CER.

Optional keys in the config section enable round-2 training: schedule (warmup + cosine), amp (bf16),
sampling (cap duplicate texts, reweight merchant groups), augment (photometric/geometric noise) and
decoder_mask_prob (VietOCR masked_language_model). Without them the trainer behaves as round 1.
"""
import argparse
import math
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.ocr.backends import package_versions, vietocr_image  # noqa: E402
from src.ocr.baseline import file_hash, read_image, read_jsonl, safe_path, write_json, write_jsonl  # noqa: E402
from src.ocr.training import augment_crop, balanced_epoch, corpus_metrics, git_state, warmup_cosine, width_batches  # noqa: E402
from src.ocr.training_data import verify_training_data  # noqa: E402


def train(config_path: Path, data: Path, out: Path, epochs: int | None = None, max_batches: int = 0,
          section: str = 'vietocr', learning_rate: float | None = None):
    import torch
    from vietocr.tool.translate import build_model, translate

    metadata = verify_training_data(data)
    settings = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    params = dict(settings[section])
    if learning_rate is not None:
        params['learning_rate'] = learning_rate
    epochs = epochs or params.get('epochs') or settings['epochs']
    if epochs < 1 or max_batches < 0:
        raise ValueError('epochs must be positive; max_batches non-negative')
    if params.get('amp') not in (None, 'bf16'):
        raise ValueError('amp must be omitted or bf16')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Training output must be new/empty')
    if params['device'].startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable')
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    cfg = yaml.safe_load(Path(params['base_config_file']).read_text())
    cfg.update(yaml.safe_load(Path(params['config_file']).read_text()))
    cfg['device'] = params['device']; cfg['cnn']['pretrained'] = False
    cfg['weights'] = str((out / 'best.pth').resolve())
    cfg['predictor']['beamsearch'] = False
    model, vocab = build_model(cfg)
    model.load_state_dict(torch.load(params['pretrained'], map_location=params['device'], weights_only=True), strict=True)
    rows = {sp: read_jsonl(data / f'{sp}.jsonl') for sp in ('train', 'val')}
    for sp in rows:
        for sample in rows[sp]:
            if set(sample['text']) - set(vocab.chars):
                raise ValueError(f"Unsupported vocabulary: {sample['sample_id']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.yml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    sampling, augment = params.get('sampling'), params.get('augment')
    epoch_size = (sampling.get('samples_per_epoch') or len(rows['train'])) if sampling else len(rows['train'])
    batches_per_epoch = math.ceil(epoch_size / params['batch_size'])
    if max_batches:
        batches_per_epoch = min(batches_per_epoch, max_batches)
    total_steps = epochs * math.ceil(batches_per_epoch / params['gradient_accumulation'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    schedule = params.get('schedule')
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine(
        step, total_steps, schedule['warmup_ratio'], schedule['min_lr_ratio'])) if schedule else None)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad, label_smoothing=params['label_smoothing'])
    use_amp = params.get('amp') == 'bf16'
    mask_prob = params.get('decoder_mask_prob', 0.0)
    history, best, stale = [], math.inf, 0
    start = perf_counter()

    def arrays(batch, rng=None):
        images = [read_image(safe_path(data, s['crop'])) for s in batch]
        if rng is not None:
            images = [augment_crop(image, rng, augment) for image in images]
        return [vietocr_image(image, cfg['dataset']) for image in images]

    def validate():
        model.eval()
        predictions = []
        # Use exact resized width buckets, as the inference adapter does, to avoid padded validation inputs.
        for batch in width_batches(rows['val'], 16):
            buckets = {}
            for sample, array in zip(batch, arrays(batch), strict=True):
                buckets.setdefault(array.shape[-1], []).append((sample, array))
            for items in buckets.values():
                tensor = torch.from_numpy(np.stack([a for _, a in items])).to(params['device'])
                sequences, _ = translate(tensor, model)
                texts = vocab.batch_decode(sequences.tolist())
                predictions.extend({'sample_id': s['sample_id'], 'ref': s['text'], 'text': text}
                                   for (s, _), text in zip(items, texts, strict=True))
        return corpus_metrics([p['ref'] for p in predictions], [p['text'] for p in predictions]), predictions

    baseline, _ = validate()
    write_json(out / 'baseline.json', baseline)
    print(f'pretrained: validation CER={baseline["cer"]:.4%} WER={baseline["wer"]:.4%}', flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rows = (balanced_epoch(rows['train'], sampling['max_text_repeats'], sampling['group_alpha'], epoch_size,
                                     settings['seed'] + epoch) if sampling else rows['train'])
        batches = width_batches(epoch_rows, params['batch_size'], settings['seed'] + epoch)[:batches_per_epoch]
        rng = np.random.default_rng([settings['seed'], epoch]) if augment else None
        total_loss, sample_count = 0., 0
        for group_start in range(0, len(batches), params['gradient_accumulation']):
            micro_batches = batches[group_start:group_start + params['gradient_accumulation']]
            optimizer.zero_grad(set_to_none=True)
            group_size = sum(len(b) for b in micro_batches)
            for batch in micro_batches:
                inputs = arrays(batch, rng)
                width = max(a.shape[-1] for a in inputs)
                padded = [np.pad(a, ((0, 0), (0, 0), (0, width - a.shape[-1])), mode='edge') for a in inputs]
                image = torch.from_numpy(np.stack(padded)).to(params['device'])
                labels = [vocab.encode(s['text']) for s in batch]
                length = max(len(s) for s in labels)
                encoded = torch.zeros((len(batch), length), dtype=torch.long, device=params['device'])
                for i, label in enumerate(labels):
                    encoded[i, :len(label)] = torch.tensor(label, device=params['device'])
                targets, decoder_input = encoded[:, 1:], encoded[:, :-1].clone()
                padding_mask = decoder_input == vocab.pad
                if mask_prob:
                    # Hide some previous characters so the decoder must read pixels, not recall memorised strings.
                    hide = (torch.rand(decoder_input.shape, device=decoder_input.device) < mask_prob) & (decoder_input > vocab.mask_token)
                    decoder_input[hide] = vocab.mask_token
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(image, decoder_input.T, padding_mask)
                loss = criterion(outputs.float().reshape(-1, len(vocab)), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss * len(batch) / group_size).backward()
                total_loss += loss.item() * len(batch); sample_count += len(batch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            if (group_start // params['gradient_accumulation']) % 25 == 0:
                print(f'epoch {epoch} batch {group_start + len(micro_batches)}/{len(batches)} loss={total_loss / sample_count:.4f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        metrics, predictions = validate()
        history.append({'epoch': epoch, 'train_loss': total_loss / sample_count, 'train_samples': sample_count,
                        'lr_end': optimizer.param_groups[0]['lr'], 'validation': metrics, 'elapsed_s': perf_counter() - start})
        print(f'epoch {epoch}: validation CER={metrics["cer"]:.4%} WER={metrics["wer"]:.4%}', flush=True)
        if metrics['cer'] < best:
            best, stale = metrics['cer'], 0
            temporary = out / 'best.pth.tmp'; torch.save(model.state_dict(), temporary); temporary.replace(out / 'best.pth')
            write_jsonl(out / 'best_predictions.jsonl', predictions)
            write_json(out / 'best.json', {'epoch': epoch, **metrics, 'weights_sha256': file_hash(out / 'best.pth')})
        else:
            stale += 1
        write_json(out / 'run.json', {'engine': 'vietocr', 'section': section, 'params': params, 'seed': settings['seed'],
                                     'epochs_requested': epochs, 'max_batches': max_batches, 'total_optimizer_steps': total_steps,
                                     'data_metadata_sha256': file_hash(data / 'metadata.json'), 'data': metadata,
                                     'pretrained_sha256': file_hash(Path(params['pretrained'])), 'versions': package_versions(),
                                     'git': git_state(ROOT), 'baseline': baseline, 'history': history,
                                     'selection': 'minimum full validation corpus NFC CER (selection-biased; test stays locked)'})
        if stale >= params['patience']:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/ocr_finetune.yaml')
    p.add_argument('--section', default='vietocr', help='Config section: vietocr (round 1) or vietocr_v2')
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed/ocr_review_v1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--learning-rate', type=float, help='Override the section learning rate (LR sweep)')
    p.add_argument('--max-batches', type=int, default=0, help='Smoke test only; 0 means full train')
    args = p.parse_args()
    train(args.config, args.data, args.out, args.epochs, args.max_batches, args.section, args.learning_rate)


if __name__ == '__main__':
    main()
