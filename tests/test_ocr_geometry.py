import cv2
import numpy as np

from src.ocr.geometry import apply_homography, assign_lines_to_regions, crop_polygon, order_quad, overlap_ratio, reading_order
from src.ocr.types import OcrLine
from src.preprocessing.opencv_rules import estimate_skew, preprocess


def line(x0, y0, x1, y1, text=""):
    return OcrLine([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text=text)


def test_overlap_ratio():
    assert overlap_ratio(line(0, 0, 10, 10).polygon, [0, 0, 10, 10]) == 1.0
    assert abs(overlap_ratio(line(0, 0, 10, 10).polygon, [5, 0, 10, 10]) - 0.5) < 1e-6
    assert overlap_ratio(line(0, 0, 10, 10).polygon, [50, 50, 5, 5]) == 0.0


def test_assign_lines_to_regions_by_majority_overlap():
    regions = [{"bbox": [0, 0, 100, 20]}, {"bbox": [0, 30, 100, 20]}]
    lines = [line(0, 2, 40, 18, "a"), line(45, 2, 90, 18, "b"), line(0, 32, 90, 48, "c"), line(0, 100, 50, 120, "outside")]
    got = assign_lines_to_regions(lines, regions)
    assert [ln.text for ln in got[0]] == ["a", "b"] and [ln.text for ln in got[1]] == ["c"]


def test_reading_order_groups_rows_then_sorts_left_to_right():
    lines = [line(60, 1, 90, 11, "r1b"), line(0, 0, 50, 10, "r1a"), line(0, 30, 50, 40, "r2")]
    assert [ln.text for ln in reading_order(lines)] == ["r1a", "r1b", "r2"]


def test_crop_polygon_rotates_vertical_crops():
    img = np.zeros((200, 200, 3), np.uint8)
    wide = crop_polygon(img, line(10, 10, 110, 30).polygon)
    tall = crop_polygon(img, line(10, 10, 30, 110).polygon)
    assert wide.shape[1] > wide.shape[0] and tall.shape[1] > tall.shape[0]   # crop dọc được xoay thành ngang


def test_diamond_boxes_keep_four_distinct_corners_and_can_be_cropped():
    polygon = [[50, 20], [80, 50], [50, 80], [20, 50]]
    ordered = order_quad(np.array(polygon)[[2, 0, 3, 1]])
    assert len(np.unique(ordered, axis=0)) == 4
    crop = crop_polygon(np.zeros((100, 100, 3), np.uint8), polygon)
    assert crop is not None and min(crop.shape[:2]) > 30


def _text_image(w=600, h=800):
    img = np.full((h, w, 3), 255, np.uint8)
    for i in range(14):
        cv2.putText(img, f"TONG TIEN {i * 1234}", (40, 80 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def test_deskew_recovers_known_tilt():
    img = _text_image()
    h, w = img.shape[:2]
    tilted = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0), (w, h), borderValue=(255, 255, 255))
    assert abs(estimate_skew(tilted) - (-5.0)) <= 0.75     # phải xoay ngược lại 5 độ
    assert abs(estimate_skew(img)) <= 0.5


def test_homography_maps_boxes_back_to_original_frame():
    img = _text_image()
    h, w = img.shape[:2]
    tilted = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0), (w, h), borderValue=(255, 255, 255))
    pre = preprocess(tilted, {"deskew": {"max_angle": 12, "min_angle": 0.5}})
    assert pre.meta["deskew"] == "applied"
    p = [[100.0, 200.0], [300.0, 200.0], [300.0, 240.0], [100.0, 240.0]]
    back = apply_homography(apply_homography(p, pre.H), np.linalg.inv(pre.H))
    assert np.allclose(back, p, atol=1e-3)


def test_preprocess_none_is_identity():
    img = _text_image()
    pre = preprocess(img, {})
    assert np.array_equal(pre.image, img) and np.allclose(pre.H, np.eye(3))
