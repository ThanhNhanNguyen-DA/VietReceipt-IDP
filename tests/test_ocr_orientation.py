import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.ocr.baseline import file_hash, write_json
from src.ocr.geometry import apply_homography, reading_order
from src.ocr.orientation import choose_orientation, load_orientation, probe_orientation, validate_rules
from src.ocr.types import OcrLine
from src.preprocessing.opencv_rules import preprocess


@pytest.fixture
def rules():
    return yaml.safe_load(Path("configs/ocr.yaml").read_text())["orientation"]


def pair(a, b, text_a="upside down", text_b="Tổng tiền"):
    return {"score_0": a, "score_180": b, "text_0": text_a, "text_180": text_b}


def test_orientation_requires_consistent_confidence_advantage(rules):
    assert choose_orientation([pair(0.6, 0.9)] * 5, rules)["degrees"] == 180
    assert choose_orientation([pair(0.9, 0.6)] * 5, rules)["degrees"] == 0
    assert choose_orientation([pair(0.8, 0.82)] * 5, rules)["degrees"] == 0
    assert choose_orientation([pair(0.3, 0.7)] * 5, rules)["degrees"] == 0
    assert choose_orientation([pair(0.6, 0.9)] * 3 + [pair(0.9, 0.8)] * 2, rules)["degrees"] == 0


def test_ambiguous_missing_or_nonfinite_evidence_keeps_original(rules):
    cases = [[], [pair(0.6, 0.9)] * 4, [pair(None, 0.9)] * 5, [pair(float("nan"), 0.9)] * 5,
             [pair(0.6, float("inf"))] * 5, [pair(-0.1, 0.9)] * 5, [pair(0.6, 1.1)] * 5,
             [pair(0.6, 0.9, text_b=" ")] * 5, [pair(0.9, 0.9)] * 5]
    for pairs in cases:
        assert choose_orientation(pairs, rules)["degrees"] == 0


@pytest.mark.parametrize("change", [{"min_lines": 0}, {"max_lines": 4}, {"min_lines": True},
                                    {"min_margin": -0.1}, {"min_aspect": 0.5}, {"min_vote_fraction": 1.1}])
def test_invalid_orientation_rules_are_rejected(rules, change):
    with pytest.raises(ValueError):
        validate_rules({**rules, **change})


def test_rotation_preserves_pixels_and_maps_pixel_centres_exactly():
    image = np.arange(30 * 50 * 3, dtype=np.uint8).reshape(30, 50, 3)
    pre = preprocess(image, {}, rotation_deg=180)
    assert np.array_equal(pre.image, image[::-1, ::-1])
    corners = [[0, 0], [49, 0], [49, 29], [0, 29]]
    assert np.allclose(apply_homography(corners, pre.H), [[49, 29], [0, 29], [0, 0], [49, 0]])
    assert np.allclose(apply_homography(apply_homography(corners, pre.H), np.linalg.inv(pre.H)), corners)
    with pytest.raises(ValueError, match="0 or 180"):
        preprocess(image, {}, rotation_deg=90)


def test_reading_order_uses_upright_frame_without_changing_output_boxes():
    H = np.array([[-1., 0., 199.], [0., -1., 299.], [0., 0., 1.]])
    upright = [([[10, 10], [70, 10], [70, 30], [10, 30]], "a"),
               ([[100, 10], [160, 10], [160, 30], [100, 30]], "b"),
               ([[10, 60], [70, 60], [70, 80], [10, 80]], "c")]
    lines = [OcrLine(apply_homography(poly, H), text=text) for poly, text in upright]
    before = [ln.polygon for ln in lines]
    assert [ln.text for ln in reading_order(lines)] == ["c", "b", "a"]
    assert [ln.text for ln in reading_order(lines, H=H)] == ["a", "b", "c"]
    assert [ln.polygon for ln in lines] == before


@pytest.fixture
def work(tmp_path):
    crop = np.zeros((20, 100, 3), np.uint8)
    crop[:10] = 255
    cv2.imwrite(str(tmp_path / "crop.png"), crop)
    cv2.imwrite(str(tmp_path / "image.png"), crop)
    write_json(tmp_path / "dataset.json", {
        "split": "val", "preproc": "none", "mode": "e2", "manifest_sha256": "manifest",
        "config": {"recognition": {"vietocr": {"batch_size": 2}}},
        "samples": [{"sample_id": f"0:e2:{i}", "mode": "e2", "image_id": "image", "crop": "crop.png",
                     "sha256": file_hash(tmp_path / "crop.png")} for i in range(5)],
        # Deliberately omit GT, regions and quality: orientation must not need them.
        "pages": [{"image_id": "image", "image_path": "image.png", "width": 100, "height": 20}],
    })
    return tmp_path


class MarkerRecognizer:
    def __init__(self):
        self.calls = 0

    def recognize(self, images):
        self.calls += 1
        return [("text", 0.6 if im[0, 0, 0] else 0.9) for im in images]


def test_probe_uses_only_e2_pixels_and_saves_replay_evidence(work, rules):
    model = MarkerRecognizer()
    report = probe_orientation(work, work, work / "orientation.json", rules, device="cpu", recognizer=model)
    page = report["pages"][0]
    assert page["degrees"] == 180 and page["valid_lines"] == 5
    assert len(page["probes"]) == 5 and model.calls == 6
    assert page["image_sha256"] == file_hash(work / "image.png")
    assert report["source_dataset_sha256"] == file_hash(work / "dataset.json")
    with pytest.raises(ValueError, match="already exists"):
        probe_orientation(work, work, work / "orientation.json", rules, recognizer=model)


def test_probe_checks_integrity_before_loading_model(work, rules):
    (work / "crop.png").write_bytes(b"changed")
    model = MarkerRecognizer()
    with pytest.raises(ValueError, match="Shared crop changed"):
        probe_orientation(work, work, work / "orientation.json", rules, recognizer=model)
    assert model.calls == 0


@pytest.mark.parametrize("change", [{"split": "test"}, {"split": "test_seen"}, {"mode": "e1"}, {"preproc": "rules"}])
def test_probe_rejects_test_or_preprocessed_input(work, rules, change):
    path = work / "dataset.json"
    write_json(path, {**json.loads(path.read_text()), **change})
    with pytest.raises(ValueError, match="train/val E2"):
        probe_orientation(work, work, work / "orientation.json", rules, recognizer=MarkerRecognizer())


def test_replay_rejects_stale_manifest_duplicate_ids_and_invalid_angles(work, rules):
    report = probe_orientation(work, work, work / "orientation.json", rules, recognizer=MarkerRecognizer())
    manifest = work / "all.jsonl"
    manifest.write_text("reference")
    path = work / "orientation.json"
    with pytest.raises(ValueError, match="manifest/split"):
        load_orientation(path, manifest, "val")
    report["manifest_sha256"] = file_hash(manifest)
    write_json(path, report)
    assert load_orientation(path, manifest, "val")["image"]["degrees"] == 180
    report["pages"] *= 2
    write_json(path, report)
    with pytest.raises(ValueError, match="Duplicate"):
        load_orientation(path, manifest, "val")
    report["pages"] = report["pages"][:1]
    report["pages"][0]["degrees"] = 90
    write_json(path, report)
    with pytest.raises(ValueError, match="0 or 180"):
        load_orientation(path, manifest, "val")
