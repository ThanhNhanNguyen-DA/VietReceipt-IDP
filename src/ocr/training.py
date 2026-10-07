"""Framework-independent batching, sampling, augmentation and corpus metrics for OCR fine-tuning."""
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.ocr.metrics import score_pair


def width_batches(rows: list[dict], batch_size: int, seed: int | None = None) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = sorted(rows, key=lambda s: s['width'] / s['height'])
    batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(batches)
    return batches


def balanced_epoch(rows: list[dict], max_text_repeats: int, group_alpha: float, size: int, seed: int) -> list[dict]:
    """Draw one epoch with replacement: cap identical (group, text) copies, then weight groups by n_g ** alpha.

    alpha=1 keeps the natural group mix of the capped pool; alpha=0 samples every group equally.
    """
    if max_text_repeats < 1 or not 0 <= group_alpha <= 1 or size < 1:
        raise ValueError("Require max_text_repeats >= 1, group_alpha in [0, 1] and size >= 1")
    rng = random.Random(seed)
    copies = defaultdict(list)
    for row in rows:
        copies[row['group_id'], row['text']].append(row)
    pools = defaultdict(list)
    for (group, _), items in sorted(copies.items()):
        pools[group].extend(rng.sample(items, min(len(items), max_text_repeats)))
    groups = sorted(pools)
    weights = [len(pools[g]) ** group_alpha for g in groups]
    return [rng.choice(pools[g]) for g in rng.choices(groups, weights=weights, k=size)]


def augment_crop(image: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Photometric/geometric noise seen on phone photos of receipts. No flips: text direction is the label."""
    if rng.random() >= cfg['prob']:
        return image
    out = image
    h, w = out.shape[:2]
    if rng.random() < cfg['pad']['prob']:
        top, bottom, left, right = (int(rng.integers(0, max(1, round(cfg['pad']['max_ratio'] * h)) + 1)) for _ in range(4))
        out = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_REPLICATE)
        h, w = out.shape[:2]
    if rng.random() < cfg['rotate']['prob']:
        angle = rng.uniform(-cfg['rotate']['max_deg'], cfg['rotate']['max_deg'])
        out = cv2.warpAffine(out, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < cfg['downscale']['prob']:
        scale = rng.uniform(cfg['downscale']['min_scale'], 1.0)
        small = cv2.resize(out, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < cfg['blur']['prob']:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, cfg['blur']['max_sigma']))
    if rng.random() < cfg['contrast']['prob']:
        alpha = rng.uniform(1 - cfg['contrast']['max_delta'], 1 + cfg['contrast']['max_delta'])
        beta = rng.uniform(-cfg['contrast']['max_shift'], cfg['contrast']['max_shift'])
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < cfg['noise']['prob']:
        noise = rng.normal(0, rng.uniform(1, cfg['noise']['max_sigma']), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < cfg['jpeg']['prob']:
        quality = int(rng.integers(cfg['jpeg']['min_quality'], 96))
        ok, encoded = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out


def warmup_cosine(step: int, total_steps: int, warmup_ratio: float, min_ratio: float) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to min_ratio at total_steps."""
    warmup = max(1, round(total_steps * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def git_state(root: Path) -> dict:
    """Commit and dirty flag of this repository, so a run can be tied to its code."""
    try:
        sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}
    return {'commit': sha, 'dirty': dirty}


def corpus_metrics(refs: list[str], hyps: list[str]) -> dict:
    if not refs or len(refs) != len(hyps):
        raise ValueError("Metrics require nonempty aligned references/predictions")
    scores = [score_pair(a, b) for a, b in zip(refs, hyps, strict=True)]
    c = sum(s['char_len_exact'] for s in scores)
    w = sum(s['word_len_exact'] for s in scores)
    if not c or not w:
        raise ValueError("Empty references cannot define CER/WER")
    return {'cer': sum(s['char_err_exact'] for s in scores) / c,
            'wer': sum(s['word_err_exact'] for s in scores) / w,
            'exact_match': sum(s['exact_match'] for s in scores) / len(scores), 'samples': len(scores)}


class PaddleCorpusMetric:
    """Plug-in for upstream RecMetric: select checkpoints using corpus NFC CER."""
    def __init__(self, main_indicator='one_minus_cer', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def reset(self):
        self.refs, self.hyps = [], []

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        for (pred, _), (ref, _) in zip(preds, labels, strict=True):
            self.refs.append(ref); self.hyps.append(pred)

    def get_metric(self):
        metrics = corpus_metrics(self.refs, self.hyps)
        self.reset()
        return {**metrics, 'acc': metrics['exact_match'], 'one_minus_cer': 1 - metrics['cer']}
