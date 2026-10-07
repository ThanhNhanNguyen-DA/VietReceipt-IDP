import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.ocr.backends import VietRecognizer, paddle_device_options, vietocr_image
from src.ocr.baseline import file_hash, fork, prepare, read_jsonl, recognize, write_json
from src.ocr.geometry import apply_homography
from src.ocr.report import evaluate, evaluation_rows
from src.preprocessing.opencv_rules import estimate_skew, find_document_quad


@pytest.fixture
def source(tmp_path):
    image = np.full((100, 160, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (100, 30), (0, 0, 0), 1)
    cv2.imwrite(str(tmp_path / "receipt.png"), image)
    regions = [{"bbox": [10, 10, 90, 20], "text": "Tổng tiền", "label": "TOTAL_COST", "role": "key"},
               {"bbox": [10, 50, 90, 20], "text": "100000", "label": "TOTAL_COST", "role": "value"},
               {"bbox": [0, 0, 2, 1], "text": "bad box", "label": "TOTAL_COST"}]
    row = {"image_id": "receipt", "image_path": "receipt.png", "height": 100, "width": 160,
           "group_id": "g1", "quality": 0.9, "split": "val", "regions": regions}
    manifest = tmp_path / "all.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    config = yaml.safe_load(Path("configs/ocr.yaml").read_text())
    config["recognition"]["paddle"]["batch_size"] = 2
    cfg = tmp_path / "ocr.yaml"
    cfg.write_text(yaml.safe_dump(config))
    return cfg, manifest, tmp_path, tmp_path / "work"


class Detector:
    def detect(self, image):
        # One GT region missed, plus a prediction outside the annotated area.
        return [([[10, 10], [100, 10], [100, 30], [10, 30]], 0.9),
                ([[110, 75], [150, 75], [150, 95], [110, 95]], 0.8)]


class Recognizer:
    def __init__(self):
        self.inputs = []

    def recognize(self, images):
        self.inputs.extend(im.copy() for im in images)
        # E1 contains two crops; E2 contains one matched and one unannotated crop.
        texts = ["Tổng tiền", "100000", "Tổng tiền", "outside"]
        start = len(self.inputs) - len(images)
        return [(s, 0.9) for s in texts[start:start + len(images)]]


def test_staged_pipeline_uses_identical_crops_and_counts_missing_regions(source):
    cfg, manifest, root, work = source
    bundle = prepare(cfg, manifest, root, work, detector=Detector())
    assert len(bundle["samples"]) == 4 and bundle["pages"][0]["skipped_regions"] == 1
    a, b = Recognizer(), Recognizer()
    recognize(work, "paddle", device="cpu", recognizer=a, batch_order="input")
    recognize(work, "vietocr", device="cpu", recognizer=b, batch_order="input")
    assert all(np.array_equal(x, y) for x, y in zip(a.inputs, b.inputs, strict=True))
    report = evaluate(work, ["paddle", "vietocr"], n_resamples=10)
    e1, e2 = report["engines"]["paddle"]["e1"]["overall"], report["engines"]["paddle"]["e2"]["overall"]
    assert e1["cer_exact"]["value"] == 0
    assert e2["region_coverage"] == 0.5 and e2["cer_exact"]["value"] == pytest.approx(6 / 15)
    roles = report["engines"]["paddle"]["e2"]["by_role"]
    assert roles["key"]["cer_exact"]["value"] == 0 and roles["value"]["cer_exact"]["value"] == 1
    assert report["engines"]["paddle"]["e2"]["by_quality"]["normal"]["regions"] == 2
    assert report["paired"]["e2"]["cer"]["diff"] == 0
    assert (work / "errors_paddle.csv").exists()
    assert json.loads((work / "report.json").read_text())["skipped_regions"] == 1


def test_modified_shared_crop_is_rejected_before_recognition(source):
    cfg, manifest, root, work = source
    bundle = prepare(cfg, manifest, root, work, mode="e1")
    crop = work / bundle["samples"][0]["crop"]
    crop.write_bytes(b"changed")
    model = Recognizer()
    with pytest.raises(ValueError, match="Shared crop changed"):
        recognize(work, "paddle", recognizer=model)
    assert model.inputs == []


def test_incomplete_or_duplicate_predictions_are_rejected(source):
    cfg, manifest, root, work = source
    bundle = prepare(cfg, manifest, root, work, mode="e1")
    preds = recognize(work, "paddle", recognizer=Recognizer())
    with pytest.raises(ValueError, match="exactly one"):
        evaluation_rows(bundle, preds[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        evaluation_rows(bundle, [*preds, preds[0]])
    assert len(read_jsonl(work / "predictions_paddle.jsonl")) == 2


def test_changed_dataset_cannot_reuse_results(source):
    cfg, manifest, root, work = source
    prepare(cfg, manifest, root, work, mode="e1")
    recognize(work, "paddle", recognizer=Recognizer())
    path = work / "dataset.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="different dataset"):
        evaluate(work, ["paddle"])


def test_test_split_is_locked_and_existing_runs_are_protected(source):
    cfg, manifest, root, work = source
    with pytest.raises(ValueError, match="locked"):
        prepare(cfg, manifest, root, work, split="test")
    prepare(cfg, manifest, root, work, mode="e1")
    with pytest.raises(ValueError, match="new/empty"):
        prepare(cfg, manifest, root, work, mode="e1")


def test_manifest_dimension_mismatch_fails(source):
    cfg, manifest, root, work = source
    row = json.loads(manifest.read_text())
    row["height"] = 101
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="dimensions"):
        prepare(cfg, manifest, root, work, mode="e1")


def test_zero_detections_count_all_reference_text_as_deletions(source):
    class EmptyDetector:
        def detect(self, image):
            return []

    cfg, manifest, root, work = source
    prepare(cfg, manifest, root, work, mode="e2", detector=EmptyDetector())
    # No model is loaded when there are no detected crops.
    assert recognize(work, "paddle", device="cpu") == []
    overall = evaluate(work, ["paddle"], n_resamples=10)["engines"]["paddle"]["e2"]["overall"]
    assert overall["cer_exact"]["value"] == 1.0
    assert overall["wer_exact"]["value"] == 1.0
    assert overall["region_coverage"] == 0.0


def test_modified_predictions_do_not_produce_a_comparison(source):
    cfg, manifest, root, work = source
    prepare(cfg, manifest, root, work, mode="e1")
    recognize(work, "paddle", recognizer=Recognizer())
    path = work / "predictions_paddle.jsonl"
    path.write_text(path.read_text().replace("100000", "200000"))
    with pytest.raises(ValueError, match="Predictions changed"):
        evaluate(work, ["paddle"])


def test_blank_page_does_not_rotate_arbitrarily():
    assert estimate_skew(np.full((100, 150, 3), 255, np.uint8)) == 0


def test_small_image_document_quad_is_in_original_coordinates():
    image = np.zeros((200, 300, 3), np.uint8)
    cv2.rectangle(image, (50, 20), (250, 180), (255, 255, 255), -1)
    quad = find_document_quad(image, 0.35, 0.97)
    assert quad is not None
    assert quad[:, 0].min() == pytest.approx(50, abs=3)
    assert quad[:, 0].max() == pytest.approx(250, abs=3)


def test_vietocr_preprocessing_preserves_rgb_and_width_limits():
    image = np.zeros((20, 100, 3), np.uint8)
    image[:, :, 2] = 255  # OpenCV red (BGR)
    dataset = {"image_height": 32, "image_min_width": 32, "image_max_width": 128}
    array = vietocr_image(image, dataset)
    assert array.shape == (3, 32, 128) and array.dtype == np.float32
    assert np.all(array[0] == 1) and np.all(array[1:] == 0)
    assert vietocr_image(image[:, :1], dataset).shape == (3, 32, 32)


def test_vietocr_width_buckets_restore_original_crop_order(monkeypatch):
    class Tensor:
        def __init__(self, array):
            self.array = array

        def to(self, device):
            return self.array

    def translate(batch, model):
        markers = np.rint(batch[:, 0, 0, 0] * 255).astype(int)
        return markers[:, None], np.full(len(markers), np.nan)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=Tensor))
    monkeypatch.setitem(sys.modules, "vietocr.tool.translate", SimpleNamespace(translate=translate))
    adapter = VietRecognizer.__new__(VietRecognizer)
    adapter.model = SimpleNamespace(device="cpu", model=None,
        config={"dataset": {"image_height": 32, "image_min_width": 32, "image_max_width": 128}},
        vocab=SimpleNamespace(batch_decode=lambda sequences: [str(s[0]) for s in sequences]))
    images = []
    for width, marker in ((20, 85), (80, 170), (20, 255)):
        image = np.zeros((20, width, 3), np.uint8)
        image[:, :, 2] = marker
        images.append(image)
    assert adapter.recognize(images) == [("85", None), ("170", None), ("255", None)]


def test_paddle_gpu_request_cannot_silently_run_on_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0))))
    with pytest.raises(ValueError, match="silent CPU fallback"):
        paddle_device_options("gpu:0")
    assert paddle_device_options("cpu") == {"device": "cpu", "enable_mkldnn": False}


def test_width_sorting_preserves_sample_to_prediction_correspondence(source):
    class WidthRecognizer:
        def __init__(self):
            self.widths = []

        def recognize(self, images):
            self.widths.extend(im.shape[1] for im in images)
            return [(str(im.shape[1]), 0.9) for im in images]

    cfg, manifest, root, work = source
    bundle = prepare(cfg, manifest, root, work, detector=Detector())
    model = WidthRecognizer()
    predictions = recognize(work, "paddle", recognizer=model, batch_order="width")
    assert model.widths == sorted(model.widths)
    assert [p["sample_id"] for p in predictions] == [s["sample_id"] for s in bundle["samples"]]
    for sample, prediction in zip(bundle["samples"], predictions, strict=True):
        assert prediction["text"] == str(cv2.imread(str(work / sample["crop"])).shape[1])


def test_rotated_pipeline_maps_detections_back_and_reads_multiline_region_upright(source):
    cfg, manifest, root, work = source
    row = json.loads(manifest.read_text())
    row["regions"] = [{"bbox": [10, 10, 100, 65], "text": "first second", "label": "ADDRESS", "region_id": 0}]
    manifest.write_text(json.dumps(row))
    H = np.array([[-1., 0., 159.], [0., -1., 99.], [0., 0., 1.]])
    polygons = [[[10, 50], [100, 50], [100, 70], [10, 70]], [[10, 10], [100, 10], [100, 30], [10, 30]]]

    class RotatedDetector:
        def detect(self, image):
            assert np.array_equal(image, cv2.imread(str(root / "receipt.png"))[::-1, ::-1])
            return [(apply_homography(p, H), 0.9) for p in polygons]

    class RotatedRecognizer:
        def recognize(self, images):
            return [("first", 0.9), ("second", 0.9)]

    decisions = root / "orientation.json"
    write_json(decisions, {"format_version": 1, "split": "val", "source_preproc": "none", "manifest_sha256": file_hash(manifest),
                           "pages": [{"image_id": row["image_id"], "image_path": row["image_path"], "width": 160, "height": 100,
                                      "degrees": 180, "image_sha256": file_hash(root / "receipt.png")}]})
    bundle = prepare(cfg, manifest, root, work, preproc="orient180", mode="e2", detector=RotatedDetector(), orientation_path=decisions)
    for ln, polygon in zip(bundle["pages"][0]["lines"], polygons, strict=True):
        assert np.allclose(ln["polygon"], polygon)
    recognize(work, "paddle", recognizer=RotatedRecognizer(), batch_order="input")
    report = evaluate(work, ["paddle"], n_resamples=10)
    assert report["engines"]["paddle"]["e2"]["overall"]["cer_exact"]["value"] == 0
    assert report["orientation"]["rotated_pages"] == [row["image_id"]]
    # Source pixels are checked before the detector or a new run can be created.
    (root / "receipt.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source image changed"):
        prepare(cfg, manifest, root, root / "new", preproc="orient180", detector=RotatedDetector(), orientation_path=decisions)
    assert not (root / "new").exists()


def test_orientation_variant_requires_explicit_decisions(source):
    cfg, manifest, root, work = source
    with pytest.raises(ValueError, match="requires --orientation"):
        prepare(cfg, manifest, root, work, preproc="orient180", mode="e1")
    with pytest.raises(ValueError, match="requires --orientation"):
        prepare(cfg, manifest, root, work, preproc="none", mode="e1", orientation_path=root / "missing.json")


def test_fork_reuses_exact_crops_without_touching_baseline_results(source):
    cfg, manifest, root, work = source
    prepare(cfg, manifest, root, work, detector=Detector())
    recognize(work, "vietocr", device="cpu", recognizer=Recognizer(), batch_order="input")
    baseline_predictions = (work / "predictions_vietocr.jsonl").read_bytes()
    weights = root / "best.pth"; weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="requires weights"):
        fork(work, root / "fork", "vietocr", {"config_file": str(cfg)})
    with pytest.raises(ValueError, match="accepts"):
        fork(work, root / "fork", "vietocr", {"model_dir": str(root)})
    bundle = fork(work, root / "fork", "vietocr", {"weights": str(weights), "config_file": None})
    original = json.loads((work / "dataset.json").read_text())
    assert bundle["samples"] == original["samples"] and bundle["pages"] == original["pages"]
    assert bundle["config"]["recognition"]["vietocr"]["weights"] == str(weights.resolve())
    assert bundle["forked_from"]["artifact_sha256"] == {"weights": file_hash(weights)}
    for sample in bundle["samples"]:
        assert file_hash(root / "fork" / sample["crop"]) == sample["sha256"]
    recognize(root / "fork", "vietocr", device="cpu", recognizer=Recognizer(), batch_order="input")
    assert evaluate(root / "fork", ["vietocr"], n_resamples=10)["engines"]["vietocr"]["e1"]["overall"]["cer_exact"]["value"] == 0
    assert (work / "predictions_vietocr.jsonl").read_bytes() == baseline_predictions
    with pytest.raises(ValueError, match="new/empty"):
        fork(work, root / "fork", "vietocr", {"weights": str(weights)})
