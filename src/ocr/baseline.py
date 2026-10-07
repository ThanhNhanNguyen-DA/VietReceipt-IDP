"""Prepare shared lossless crops, then recognize them in separate environments."""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from PIL import Image

from src.ocr.backends import package_versions
from src.ocr.geometry import apply_homography, crop_polygon
from src.ocr.types import OcrLine, PageResult
from src.preprocessing.opencv_rules import preprocess


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data directory: {relative}")
    return path


def read_image(path: Path):
    # Disable implicit EXIF rotation: manifest boxes are in encoded image coordinates.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def valid_regions(record: dict, min_side: float) -> tuple[list[dict], int]:
    regions = []
    for i, reg in enumerate(record["regions"]):
        _, _, w, h = reg["bbox"]
        if min(w, h) >= min_side and reg["text"].strip():
            regions.append({**reg, "region_id": i})
    return regions, len(record["regions"]) - len(regions)


def prepare(config_path: Path, manifest: Path, data_root: Path, out: Path, split="val", preproc="none",
            mode="both", limit=0, allow_test=False, device="gpu:0", detector=None, orientation_path: Path | None = None) -> dict:
    if split in ("test", "test_seen") and not allow_test:
        raise ValueError("Test is locked. Use --allow-test only for the final evaluation.")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if split not in cfg["evaluation"]["splits_with_manifest"]:
        raise ValueError(f"Unknown split: {split}")
    steps = cfg["preprocessing"]["variants"][preproc]
    records = [r for r in read_jsonl(manifest) if r["split"] == split]
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError(f"No records in split {split}")
    orientations = {}
    if (preproc == "orient180") != (orientation_path is not None):
        raise ValueError("orient180 requires --orientation; other variants must omit it")
    if orientation_path is not None:
        from src.ocr.orientation import load_orientation

        orientations = load_orientation(orientation_path, manifest, split)
        for record in records:
            decision = orientations.get(record["image_id"])
            if (decision is None or decision["image_path"] != record["image_path"]
                    or (decision["width"], decision["height"]) != (record["width"], record["height"])
                    or decision["image_sha256"] != file_hash(safe_path(data_root, record["image_path"]))):
                raise ValueError(f"Orientation missing or source image changed: {record['image_id']}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be a new/empty directory: {out}")
    if mode not in ("e1", "e2", "both"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode != "e1" and detector is None:
        from src.ocr.backends import PaddleDetector

        detector = PaddleDetector(cfg["detection"], device)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    pages, samples = [], []

    def save_crop(image, polygon, sample):
        crop = crop_polygon(image, polygon, **cfg["crops"])
        if crop is None:
            raise ValueError(f"Degenerate crop: {sample['sample_id']}")
        relative = f"crops/{len(samples):07d}.png"
        path = out / relative
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"Cannot write {path}")
        samples.append({**sample, "crop": relative, "sha256": file_hash(path)})

    for index, record in enumerate(records):
        start = perf_counter()
        image = read_image(safe_path(data_root, record["image_path"]))
        if image.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Image dimensions do not match manifest: {record['image_id']}")
        t0 = perf_counter()
        decision = orientations.get(record["image_id"])
        pre = preprocess(image, steps, rotation_deg=decision["degrees"] if decision else 0)
        if decision:
            pre.meta.update(orientation={k: v for k, v in decision.items() if k != "probes"}, H=pre.H.tolist())
        pre_time = perf_counter() - t0
        regs, skipped = valid_regions(record, cfg["evaluation"]["region_match"]["min_gt_side"])
        page = {"image_id": record["image_id"], "group_id": record["group_id"], "quality": record["quality"],
                "image_path": record["image_path"], "width": record["width"], "height": record["height"],
                "regions": regs, "skipped_regions": skipped, "lines": [], "meta": pre.meta,
                "timing_s": {"preprocessing": pre_time}}
        if mode != "e2":
            for reg in regs:
                x, y, w, h = reg["bbox"]
                poly = reg.get("polygon", [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                save_crop(pre.image, apply_homography(poly, pre.H),
                          {"sample_id": f"{index}:e1:{reg['region_id']}", "image_id": record["image_id"],
                           "mode": "e1", "region_id": reg["region_id"]})
        if mode != "e1":
            t0 = perf_counter()
            detected = detector.detect(pre.image)
            page["timing_s"]["detection"] = perf_counter() - t0
            inverse = np.linalg.inv(pre.H)
            for poly, score in detected:
                if min(cv2.minAreaRect(np.asarray(poly, dtype=np.float32))[1]) < cfg["detection"]["min_box_side"]:
                    continue
                line_id = len(page["lines"])
                sid = f"{index}:e2:{line_id}"
                save_crop(pre.image, poly, {"sample_id": sid, "image_id": record["image_id"], "mode": "e2", "line_id": line_id})
                page["lines"].append({"sample_id": sid, "polygon": apply_homography(poly, inverse), "det_score": score})
        page["timing_s"]["prepare_total"] = perf_counter() - start
        pages.append(page)
        print(f"prepared {index + 1}/{len(records)}: {record['image_id']}", flush=True)
    bundle = {"format_version": 1, "split": split, "preproc": preproc, "mode": mode, "limit": limit,
              "config": cfg, "manifest_sha256": file_hash(manifest), "device_detection": device if mode != "e1" else None,
              "versions_prepare": package_versions(), "pages": pages, "samples": samples}
    if orientation_path:
        bundle["orientation_sha256"] = file_hash(orientation_path)
        bundle["orientation"] = json.loads(orientation_path.read_text(encoding="utf-8"))
    write_json(out / "dataset.json", bundle)
    return bundle


# Keys a fork may override; anything else would change crops or references, not just the recognizer.
FORK_OVERRIDES = {"vietocr": ("config_file", "weights"), "paddle": ("model_dir",)}


def _artifact_hash(path: Path) -> str:
    if path.is_file():
        return file_hash(path)
    files = sorted(f for f in path.rglob("*") if f.is_file())
    if not files:
        raise ValueError(f"Model artifact is empty or missing: {path}")
    return hashlib.sha256("".join(f"{f.relative_to(path)}:{file_hash(f)}\n" for f in files).encode()).hexdigest()


def fork(work: Path, out: Path, engine: str, overrides: dict) -> dict:
    """New run over the exact same crops/references with another recognizer checkpoint.

    Crops are hard-linked (copied across filesystems) so baseline predictions in `work` are never overwritten.
    """
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides or set(overrides) - set(FORK_OVERRIDES[engine]):
        raise ValueError(f"{engine} fork accepts {FORK_OVERRIDES[engine]}")
    if engine == "vietocr" and "weights" not in overrides:
        raise ValueError("VietOCR fork requires weights so the checkpoint itself is hashed")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be a new/empty directory: {out}")
    source = work / "dataset.json"
    bundle = json.loads(source.read_text(encoding="utf-8"))
    for sample in bundle["samples"]:
        if file_hash(safe_path(work, sample["crop"])) != sample["sha256"]:
            raise ValueError(f"Shared crop changed: {sample['sample_id']}")
    artifacts = {k: _artifact_hash(Path(v)) for k, v in overrides.items()}
    out.mkdir(parents=True, exist_ok=True)
    for sample in bundle["samples"]:
        src, dst = safe_path(work, sample["crop"]), safe_path(out, sample["crop"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    forked = copy.deepcopy(bundle)
    forked["config"]["recognition"][engine].update({k: str(Path(v).resolve()) for k, v in overrides.items()})
    forked["forked_from"] = {"work": str(work), "dataset_sha256": file_hash(source), "engine": engine,
                             "overrides": forked["config"]["recognition"][engine], "artifact_sha256": artifacts}
    write_json(out / "dataset.json", forked)
    return forked


def recognize(work: Path, engine: str, device: str | None = None, recognizer=None, batch_order="width") -> list[dict]:
    bundle = json.loads((work / "dataset.json").read_text(encoding="utf-8"))
    cfg = bundle["config"]["recognition"][engine]
    device = device or ("gpu:0" if engine == "paddle" else cfg["device"])
    batch_size = cfg["batch_size"]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if batch_order not in ("width", "input"):
        raise ValueError("batch_order must be width or input")
    # Verify the entire shared crop set before loading a model or replacing results.
    ratios = {}
    for sample in bundle["samples"]:
        path = safe_path(work, sample["crop"])
        if file_hash(path) != sample["sha256"]:
            raise ValueError(f"Shared crop changed: {sample['sample_id']}")
        if batch_order == "width":
            with Image.open(path) as image:
                ratios[sample["sample_id"]] = image.width / image.height
    samples = (sorted(bundle["samples"], key=lambda s: ratios[s["sample_id"]])
               if batch_order == "width" else bundle["samples"])
    if recognizer is None and bundle["samples"]:
        from src.ocr.backends import PaddleRecognizer, VietRecognizer

        recognizer = (PaddleRecognizer if engine == "paddle" else VietRecognizer)(cfg, device)
    predictions, batch_times = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [read_image(safe_path(work, sample["crop"])) for sample in batch]
        t0 = perf_counter()
        results = recognizer.recognize(images)
        elapsed = perf_counter() - t0
        if len(results) != len(batch):
            raise ValueError("Recognizer returned an unexpected number of predictions")
        batch_times.append(elapsed)
        for sample, (text, score) in zip(batch, results, strict=True):
            predictions.append({"sample_id": sample["sample_id"], "text": text, "rec_score": score,
                                "timing_s": elapsed / len(batch)})
        if len(predictions) % 256 == 0 or len(predictions) == len(samples):
            print(f"recognized {len(predictions)}/{len(samples)} ({engine})", flush=True)
    by_id = {p["sample_id"]: p for p in predictions}
    predictions = [by_id[s["sample_id"]] for s in bundle["samples"]]
    # Atomic publication keeps an interrupted run from looking complete.
    target = work / f"predictions_{engine}.jsonl"
    temporary = target.with_suffix(".tmp")
    write_jsonl(temporary, predictions)
    temporary.replace(target)
    by_id = {p["sample_id"]: p for p in predictions}
    pages = []
    for page in bundle["pages"]:
        lines = [OcrLine(ln["polygon"], ln["det_score"], by_id[ln["sample_id"]]["text"], by_id[ln["sample_id"]]["rec_score"])
                 for ln in page["lines"]]
        if bundle["mode"] != "e1":
            result = PageResult(page["image_id"], bundle["preproc"], lines, page["timing_s"],
                                {**page["meta"], "engine": engine, "image_path": page["image_path"],
                                 "width": page["width"], "height": page["height"]})
            pages.append(result.to_dict())
    write_jsonl(work / f"pages_{engine}.jsonl", pages)
    write_json(work / f"run_{engine}.json", {"engine": engine, "device": device, "config": cfg, "batch_order": batch_order,
               "dataset_sha256": file_hash(work / "dataset.json"), "predictions_sha256": file_hash(target), "versions": package_versions(),
               "recognition_total_s": sum(batch_times), "batch_times_s": batch_times,
               "timing_note": "Inference wall time; model initialization and file I/O excluded. Per-crop time is amortized by batch."})
    return predictions
