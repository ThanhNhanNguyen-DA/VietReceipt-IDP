"""Versioned OCR crop preparation; source annotations and held-out test stay untouched."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from src.ocr.baseline import file_hash, read_image, read_jsonl, safe_path, write_json, write_jsonl
from src.ocr.metrics import norm


def prepare_audit(manifest: Path, data_root: Path, recognition_dir: Path, config: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Audit output must be new/empty")
    records = {r['image_id']: r for r in read_jsonl(manifest) if r['split'] in ('train', 'val')}
    groups = {sp: {r['group_id'] for r in records.values() if r['split'] == sp} for sp in ('train', 'val')}
    if groups['train'] & groups['val']:
        raise ValueError("Train/validation merchant groups overlap")
    source = {}
    with (data_root / 'raw/mcocr/mcocr_train_df.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row['img_id'] in records:
                source[row['img_id']] = row['anno_texts'].split('|||'), row['anno_labels'].split('|||')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'crops').mkdir(exist_ok=True)
    samples, ids, issues, pixel_groups = [], set(), [], defaultdict(list)
    for split in ('train', 'val'):
        for line in (recognition_dir / f'recognition_{split}.tsv').read_text(encoding='utf-8').splitlines():
            relative, separator, text = line.partition('\t')
            if not separator or not norm(text):
                raise ValueError("Empty or malformed crop label")
            sample_id = Path(relative).stem
            stem, index = sample_id.rsplit('_', 1)
            image_id, source_index = stem + '.jpg', int(index)
            record = records.get(image_id)
            if record is None or record['split'] != split or sample_id in ids:
                raise ValueError(f"Wrong split, unknown image or duplicate ID: {sample_id}")
            ids.add(sample_id)
            texts, labels = source[image_id]
            if norm(texts[source_index]) != norm(text):
                raise ValueError(f"Crop transcription differs from source CSV: {sample_id}")
            path = safe_path(data_root, relative)
            image = read_image(path)
            h, w = image.shape[:2]
            pixel_hash = hashlib.sha256(f'{w}x{h}'.encode() + image.tobytes()).hexdigest()
            target = out / 'crops' / f'{sample_id}.png'
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Cannot save crop: {target}")
            sample = {'sample_id': sample_id, 'image_id': image_id, 'group_id': record['group_id'], 'split': split,
                      'source_index': source_index, 'source_label': labels[source_index], 'text': norm(text),
                      'source_crop': relative, 'source_sha256': file_hash(path), 'pixel_sha256': pixel_hash,
                      'crop': str(target.relative_to(out)), 'sha256': file_hash(target), 'width': w, 'height': h, 'mode': 'e1'}
            samples.append(sample)
            pixel_groups[pixel_hash].append(sample)
            if min(w, h) < 8 or len(sample['text']) > 100:
                issues.append({'sample_id': sample_id, 'reason': 'invalid_shape_or_text_length'})
    for duplicates in pixel_groups.values():
        if len(duplicates) > 1:
            reason = 'cross_split_duplicate' if len({s['split'] for s in duplicates}) > 1 else 'duplicate_pixels'
            if len({s['text'] for s in duplicates}) > 1:
                reason = 'conflicting_transcription'
            issues.extend({'sample_id': s['sample_id'], 'reason': reason} for s in duplicates)
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    bundle = {'format_version': 1, 'split': 'train+val', 'preproc': 'provided_upright_crops', 'mode': 'e1',
              'manifest_sha256': file_hash(manifest), 'config': cfg, 'pages': [], 'samples': samples,
              'annotation_sha256': {sp: file_hash(recognition_dir / f'recognition_{sp}.tsv') for sp in ('train', 'val')},
              'issues': issues, 'scope': 'Structural audit of provided recognition crops; no test crops read. Not a complete manual review.'}
    write_json(out / 'dataset.json', bundle)
    write_json(out / 'audit.json', {'images': {sp: sum(r['split'] == sp for r in records.values()) for sp in ('train', 'val')},
                                   'crops': {sp: sum(s['split'] == sp for s in samples) for sp in ('train', 'val')},
                                   'issues': issues, 'group_overlap': [], 'scope': bundle['scope']})
    return bundle


def finalize_data(work: Path, corrections: Path, out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training output must be new/empty")
    bundle = json.loads((work / 'dataset.json').read_text(encoding='utf-8'))
    edits = json.loads(corrections.read_text(encoding='utf-8'))
    if edits['source_manifest_sha256'] != bundle['manifest_sha256']:
        raise ValueError("Corrections belong to a different manifest")
    by_id = {s['sample_id']: s for s in bundle['samples']}
    patches = {}
    for patch in edits['corrections']:
        sid = patch['sample_id']
        if sid in patches or sid not in by_id or patch['old_text'] != by_id[sid]['text']:
            raise ValueError(f"Duplicate, unknown or stale correction: {sid}")
        if patch['action'] not in ('correct_text', 'rotate', 'exclude') or not patch.get('evidence'):
            raise ValueError("Corrections require an action and visual evidence")
        if patch['action'] == 'correct_text' and not norm(patch['new_text']):
            raise ValueError("Corrected text must not be empty")
        if type(patch.get('rotation_deg', 0)) is not int or patch.get('rotation_deg', 0) not in (0, 180):
            raise ValueError("Reviewed rotation must be 0 or 180")
        if patch['action'] == 'rotate' and patch.get('rotation_deg') != 180:
            raise ValueError("rotate action requires 180 degrees")
        patches[sid] = patch
    if bundle['issues']:
        unresolved = {s['sample_id'] for s in bundle['issues']} - {sid for sid, p in patches.items() if p['action'] == 'exclude'}
        if unresolved:
            raise ValueError(f"Structural issues require review/exclusion: {sorted(unresolved)[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'crops').mkdir(exist_ok=True)
    selected, excluded = [], []
    for sample in bundle['samples']:
        patch = patches.get(sample['sample_id'])
        if patch and patch['action'] == 'exclude':
            excluded.append(sample['sample_id'])
            continue
        path = safe_path(work, sample['crop'])
        if file_hash(path) != sample['sha256']:
            raise ValueError(f"Audited crop changed: {sample['sample_id']}")
        target = out / sample['crop']
        if patch and patch.get('rotation_deg'):
            image = cv2.rotate(read_image(path), cv2.ROTATE_180)
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Cannot write corrected crop: {target}")
            pixel_hash = hashlib.sha256(f"{sample['width']}x{sample['height']}".encode() + image.tobytes()).hexdigest()
        else:
            target.write_bytes(path.read_bytes())
            pixel_hash = sample['pixel_sha256']
        selected.append({**sample, 'text': norm(patch['new_text']) if patch and patch['action'] == 'correct_text' else sample['text'],
                         'sha256': file_hash(target), 'pixel_sha256': pixel_hash, 'reviewed': bool(patch),
                         'review_rotation_deg': patch.get('rotation_deg', 0) if patch else 0})
    for sp in ('train', 'val'):
        subset = [s for s in selected if s['split'] == sp]
        if not subset:
            raise ValueError(f"No {sp} crops")
        write_jsonl(out / f'{sp}.jsonl', subset)
        (out / f'{sp}.tsv').write_text(''.join(f"{s['crop']}\t{s['text']}\n" for s in subset), encoding='utf-8')
    metadata = {'format_version': 1, 'source_manifest_sha256': bundle['manifest_sha256'],
                'audit_dataset_sha256': file_hash(work / 'dataset.json'), 'corrections_sha256': file_hash(corrections),
                'counts': {sp: sum(s['split'] == sp for s in selected) for sp in ('train', 'val')}, 'excluded': excluded,
                'files_sha256': {f: file_hash(out / f) for f in ('train.jsonl', 'val.jsonl', 'train.tsv', 'val.tsv')},
                'scope': 'Structurally audited train/val crops plus explicitly reviewed corrections; remaining labels are source annotations.'}
    write_json(out / 'metadata.json', metadata)
    return metadata


def verify_training_data(root: Path) -> dict:
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    for filename, expected in metadata['files_sha256'].items():
        if file_hash(root / filename) != expected:
            raise ValueError(f"Training annotations changed: {filename}")
    ids, groups, pixels = {}, {}, {}
    for split in ('train', 'val'):
        rows = read_jsonl(root / f'{split}.jsonl')
        if len(rows) != metadata['counts'][split]:
            raise ValueError("Training counts differ from metadata")
        ids[split], groups[split], pixels[split] = set(), set(), set()
        for sample in rows:
            if sample['split'] != split or sample['sample_id'] in ids[split]:
                raise ValueError("Duplicate IDs or unexpected split")
            ids[split].add(sample['sample_id']); groups[split].add(sample['group_id']); pixels[split].add(sample['pixel_sha256'])
            if file_hash(safe_path(root, sample['crop'])) != sample['sha256']:
                raise ValueError(f"Training crop changed: {sample['sample_id']}")
    if ids['train'] & ids['val'] or groups['train'] & groups['val'] or pixels['train'] & pixels['val']:
        raise ValueError("Train/validation leakage")
    return metadata
