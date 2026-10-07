import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ocr.baseline import file_hash, read_jsonl, write_json, write_jsonl
from src.ocr.training import PaddleCorpusMetric, augment_crop, balanced_epoch, corpus_metrics, warmup_cosine, width_batches
from src.ocr.training_data import finalize_data, prepare_audit, verify_training_data


@pytest.fixture
def audited(tmp_path):
    raw = tmp_path / 'raw/mcocr'; raw.mkdir(parents=True)
    source = []
    records = []
    for i, sp in enumerate(('train', 'val')):
        image_id = f'example_{i}.jpg'
        crop_path = tmp_path / f'crop_{i}.png'
        image = np.full((20, 80, 3), i * 50, np.uint8)
        image[:10] = 200
        cv2.imwrite(str(crop_path), image)
        name = f'example_{i}_0.png'
        crop_path.rename(tmp_path / name)
        (tmp_path / f'recognition_{sp}.tsv').write_text(f'{name}\tTổng tiền\n')
        source.append({'img_id': image_id, 'anno_texts': 'Tổng tiền', 'anno_labels': 'TOTAL_COST'})
        records.append({'image_id': image_id, 'split': sp, 'group_id': str(i)})
    with (raw / 'mcocr_train_df.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=source[0].keys()); writer.writeheader(); writer.writerows(source)
    manifest = tmp_path / 'all.jsonl'; write_jsonl(manifest, records)
    out = tmp_path / 'audit'
    prepare_audit(manifest, tmp_path, tmp_path, Path('configs/ocr.yaml'), out)
    corrections = tmp_path / 'corrections.json'
    write_json(corrections, {'source_manifest_sha256': file_hash(manifest), 'corrections': []})
    return tmp_path, out, corrections


def test_versioned_corrections_preserve_source_and_verify_pixels(audited):
    root, audit, corrections = audited
    source_hash = file_hash(audit / 'crops/example_0_0.png')
    patch = json.loads(corrections.read_text())
    patch['corrections'] = [{'sample_id': 'example_0_0', 'old_text': 'Tổng tiền', 'new_text': 'TỔNG TIỀN',
                             'action': 'correct_text', 'rotation_deg': 180, 'evidence': 'Inspected crop'}]
    write_json(corrections, patch)
    target = root / 'version'
    metadata = finalize_data(audit, corrections, target)
    assert metadata['counts'] == {'train': 1, 'val': 1}
    assert file_hash(audit / 'crops/example_0_0.png') == source_hash
    assert read_jsonl(target / 'train.jsonl')[0]['text'] == 'TỔNG TIỀN'
    assert np.array_equal(cv2.imread(str(target / 'crops/example_0_0.png')), cv2.imread(str(audit / 'crops/example_0_0.png'))[::-1, ::-1])
    verify_training_data(target)
    (target / 'crops/example_0_0.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='crop changed'):
        verify_training_data(target)


def test_stale_corrections_cannot_silently_relabel(audited):
    root, audit, corrections = audited
    patch = json.loads(corrections.read_text())
    patch['corrections'] = [{'sample_id': 'example_0_0', 'old_text': 'wrong', 'new_text': 'new',
                             'action': 'correct_text', 'evidence': 'Inspected'}]
    write_json(corrections, patch)
    with pytest.raises(ValueError, match='stale correction'):
        finalize_data(audit, corrections, root / 'version')
    assert not (root / 'version').exists()


def test_audit_rejects_group_leakage_before_copying(audited):
    root, _, _ = audited
    manifest = root / 'all.jsonl'; rows = read_jsonl(manifest)
    rows[1]['group_id'] = rows[0]['group_id']; write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match='groups overlap'):
        prepare_audit(manifest, root, root, Path('configs/ocr.yaml'), root / 'new')
    assert not (root / 'new').exists()


def test_edited_training_tsv_is_rejected(audited):
    root, audit, corrections = audited
    target = root / 'version'; finalize_data(audit, corrections, target)
    (target / 'train.tsv').write_text('new labels')
    with pytest.raises(ValueError, match='annotations changed'):
        verify_training_data(target)


def test_width_batches_keep_all_samples_including_partial_batch():
    rows = [{'width': i + 1, 'height': 1, 'id': i} for i in range(11)]
    batches = width_batches(rows, 4, 42)
    assert sorted(r['id'] for b in batches for r in b) == list(range(11))
    assert sorted(map(len, batches)) == [3, 4, 4]
    assert width_batches(rows, 4, 42) == batches


def test_corpus_metric_is_length_weighted_and_keeps_accents_case_spaces():
    m = corpus_metrics(['abcdefghij', 'á'], ['abcdefghij', 'a'])
    assert m['cer'] == pytest.approx(1 / 11)
    assert m['wer'] == .5
    metric = PaddleCorpusMetric()
    metric(([('AB', .9)], [('ab', 1)]))
    m = metric.get_metric()
    assert m['cer'] == 1 and m['one_minus_cer'] == 0
    assert metric.refs == []


AUGMENT = {'prob': 1.0, 'pad': {'prob': 1, 'max_ratio': 0.2}, 'rotate': {'prob': 1, 'max_deg': 2},
           'downscale': {'prob': 1, 'min_scale': 0.5}, 'blur': {'prob': 1, 'max_sigma': 1},
           'contrast': {'prob': 1, 'max_delta': 0.3, 'max_shift': 20}, 'noise': {'prob': 1, 'max_sigma': 5},
           'jpeg': {'prob': 1, 'min_quality': 40}}


def test_balanced_epoch_caps_duplicate_texts_and_lifts_small_groups():
    rows = [{'group_id': 'big', 'text': 'same', 'id': i} for i in range(90)]
    rows += [{'group_id': 'big', 'text': f'unique {i}', 'id': 100 + i} for i in range(8)]
    rows += [{'group_id': 'small', 'text': f'small {i}', 'id': 200 + i} for i in range(2)]
    epoch = balanced_epoch(rows, max_text_repeats=2, group_alpha=0.5, size=2000, seed=1)
    assert len(epoch) == 2000
    assert len({r['id'] for r in epoch if r['text'] == 'same'}) == 2
    # Pool sizes 10 vs 2 -> P(small) = sqrt(2) / (sqrt(10) + sqrt(2)) ~ 0.31 instead of 2% of rows.
    assert 0.25 < sum(r['group_id'] == 'small' for r in epoch) / 2000 < 0.37
    assert balanced_epoch(rows, 2, 0.5, 50, seed=1) == balanced_epoch(rows, 2, 0.5, 50, seed=1)
    with pytest.raises(ValueError):
        balanced_epoch(rows, 0, 0.5, 10, seed=1)


def test_augmentation_keeps_uint8_bgr_and_is_reproducible():
    image = np.full((24, 120, 3), 230, np.uint8); cv2.putText(image, 'Tong 100', (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    a = augment_crop(image, np.random.default_rng([42, 1]), AUGMENT)
    b = augment_crop(image, np.random.default_rng([42, 1]), AUGMENT)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
    assert a.shape[0] >= 24 and a.shape[1] >= 120 and not np.array_equal(a[:24, :120], image)
    assert augment_crop(image, np.random.default_rng(0), {**AUGMENT, 'prob': 0}) is image


def test_warmup_cosine_schedule_bounds():
    values = [warmup_cosine(s, 100, 0.1, 0.05) for s in range(100)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in itertools.pairwise(values[9:]))
    assert values[-1] == pytest.approx(0.05, abs=1e-3)
