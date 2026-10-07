"""Probe page orientation using detected crops and recognizer confidence, never GT text."""
import json
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.ocr.backends import VietRecognizer, package_versions
from src.ocr.baseline import file_hash, read_image, safe_path, write_json


def choose_orientation(pairs: list[dict], rules: dict) -> dict:
    """Conservative page vote. Scores are heuristics, not calibrated probabilities."""
    valid = [p for p in pairs if p["text_0"].strip() and p["text_180"].strip()
             and all(isinstance(p[k], (int, float)) and math.isfinite(p[k]) and 0 <= p[k] <= 1
                     for k in ("score_0", "score_180"))]
    result = {"degrees": 0, "valid_lines": len(valid), "reason": "insufficient_evidence",
              "median_score_0": None, "median_score_180": None, "median_margin": None, "vote_fraction_180": None}
    if not valid:
        return result
    result.update(median_score_0=float(np.median([p["score_0"] for p in valid])),
                  median_score_180=float(np.median([p["score_180"] for p in valid])),
                  median_margin=float(np.median([p["score_180"] - p["score_0"] for p in valid])),
                  vote_fraction_180=sum(p["score_180"] > p["score_0"] for p in valid) / len(valid))
    if len(valid) >= rules["min_lines"]:
        rotate = (result["median_margin"] >= rules["min_margin"]
                  and result["vote_fraction_180"] >= rules["min_vote_fraction"]
                  and result["median_score_180"] >= rules["min_rotated_score"])
        result.update(degrees=180 if rotate else 0, reason="rotated_confidence_vote" if rotate else "keep_original")
    return result


def validate_rules(rules: dict):
    for key in ("min_lines", "max_lines", "min_crop_side"):
        if type(rules[key]) is not int or rules[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if rules["min_lines"] > rules["max_lines"]:
        raise ValueError("min_lines must not exceed max_lines")
    for key in ("min_margin", "min_vote_fraction", "min_rotated_score"):
        if not isinstance(rules[key], (int, float)) or not 0 <= rules[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 1 <= rules["min_aspect"] <= rules["max_aspect"]:
        raise ValueError("Require 1 <= min_aspect <= max_aspect")


def probe_orientation(work: Path, data_root: Path, out: Path, rules: dict, device="cuda:0", recognizer=None) -> dict:
    validate_rules(rules)
    if out.exists():
        raise ValueError(f"Orientation output already exists: {out}")
    source = work / "dataset.json"
    bundle = json.loads(source.read_text(encoding="utf-8"))
    if bundle["split"] not in ("train", "val") or bundle["preproc"] != "none" or bundle["mode"] == "e1":
        raise ValueError("Orientation probe requires a train/val E2 run with preprocessing none")
    cfg = bundle["config"]["recognition"]["vietocr"]
    batch_size = cfg["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    candidates, pages = [], []
    # Hash every detected crop before loading the model; do not inspect E1/GT.
    detected = {}
    for sample in bundle["samples"]:
        if sample["mode"] != "e2":
            continue
        path = safe_path(work, sample["crop"])
        if file_hash(path) != sample["sha256"]:
            raise ValueError(f"Shared crop changed: {sample['sample_id']}")
        image = read_image(path)
        h, w = image.shape[:2]
        if min(h, w) >= rules["min_crop_side"] and rules["min_aspect"] <= w / h <= rules["max_aspect"]:
            detected.setdefault(sample["image_id"], []).append((sample, w / h, w * h))
    for page in bundle["pages"]:
        image_path = safe_path(data_root, page["image_path"])
        pages.append({"image_id": page["image_id"], "image_path": page["image_path"],
                      "width": page["width"], "height": page["height"], "image_sha256": file_hash(image_path), "probes": []})
        selected = sorted(detected.get(page["image_id"], []), key=lambda s: (-s[2], s[0]["sample_id"]))[:rules["max_lines"]]
        candidates.extend((len(pages) - 1, sample, ratio) for sample, ratio, _ in selected)
    candidates.sort(key=lambda s: s[2])
    if candidates and recognizer is None:
        recognizer = VietRecognizer(cfg, device)
    inference_s = 0.0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        images = [read_image(safe_path(work, s["crop"])) for _, s, _ in batch]
        t0 = perf_counter()
        upright = recognizer.recognize(images)
        rotated = recognizer.recognize([cv2.rotate(im, cv2.ROTATE_180) for im in images])
        inference_s += perf_counter() - t0
        if len(upright) != len(batch) or len(rotated) != len(batch):
            raise ValueError("Recognizer returned an unexpected number of orientation probes")
        for (page_index, sample, _), a, b in zip(batch, upright, rotated, strict=True):
            pages[page_index]["probes"].append({"sample_id": sample["sample_id"], "crop_sha256": sample["sha256"],
                                                "text_0": a[0], "score_0": a[1], "text_180": b[0], "score_180": b[1]})
        print(f"orientation probes {min(start + batch_size, len(candidates))}/{len(candidates)}", flush=True)
    for page in pages:
        page.update(choose_orientation(page["probes"], rules))
    report = {"format_version": 1, "split": bundle["split"], "source_preproc": bundle["preproc"],
              "source_dataset_sha256": file_hash(source), "manifest_sha256": bundle["manifest_sha256"],
              "engine": "vietocr", "config": cfg, "device": device, "versions": package_versions(), "rules": rules,
              "inference_s": inference_s, "pages": pages,
              "note": "Selection uses only detected crop pixels and recognition confidence; no GT text, labels, roles or quality."}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    write_json(temporary, report)
    temporary.replace(out)
    print(f"selected 180 degrees for {sum(p['degrees'] == 180 for p in pages)}/{len(pages)} pages", flush=True)
    return report


def load_orientation(path: Path, manifest: Path, split: str) -> dict[str, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report["format_version"] != 1 or report["manifest_sha256"] != file_hash(manifest)
            or report["split"] != split or report["source_preproc"] != "none"):
        raise ValueError("Orientation decisions do not match this manifest/split or source preprocessing")
    pages = report["pages"]
    if len({p["image_id"] for p in pages}) != len(pages):
        raise ValueError("Duplicate orientation image IDs")
    if any(type(p["degrees"]) is not int or p["degrees"] not in (0, 180) for p in pages):
        raise ValueError("Orientation degrees must be 0 or 180")
    return {p["image_id"]: p for p in pages}
