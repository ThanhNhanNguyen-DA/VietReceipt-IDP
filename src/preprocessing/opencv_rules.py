"""Tiền xử lý OpenCV rule-based (kế hoạch mục 4.2): perspective (tuỳ chọn), deskew, CLAHE, khử nhiễu.

Mọi phép hình học gộp thành MỘT homography H (ảnh gốc -> ảnh đã xử lý) để đưa box OCR về toạ độ ảnh gốc,
nơi nhãn GT nằm; phép quang học (CLAHE, denoise) không đổi toạ độ. Mỗi bước an toàn: không đủ căn cứ thì bỏ qua và ghi lý do vào meta.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.ocr.geometry import order_quad


@dataclass
class Preprocessed:
    image: np.ndarray                       # BGR
    H: np.ndarray                           # 3x3, toạ độ gốc -> toạ độ ảnh đã xử lý
    meta: dict = field(default_factory=dict)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _rotation_homography(angle_deg: float, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Xoay quanh tâm, mở rộng canvas để không cắt mất góc. Góc dương = ngược chiều kim đồng hồ (quy ước của OpenCV)."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = round(h * sin + w * cos), round(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return np.vstack([m, [0, 0, 1]]), (nw, nh)


def estimate_skew(img: np.ndarray, max_angle: float = 12.0, step: float = 0.25) -> float:
    """Góc nghiêng (độ) bằng cực đại phương sai của histogram chiếu ngang, thu nhỏ để nhanh.

    Trả về góc cần xoay (quy ước OpenCV) để dòng chữ nằm ngang. Chỉ nhìn các pixel tối (chữ), nên ít bị nền ảnh chi phối.
    """
    g = _gray(img)
    scale = 600 / max(g.shape)
    if scale < 1:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = bw.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        r = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST)
        s = float(np.var(r.sum(axis=1, dtype=np.float64)))
        if s > best_s or (s == best_s and abs(a) < abs(best_a)):
            best_a, best_s = float(a), s
    # xoay ảnh nhị phân bằng best_a làm hàng chữ ngang nhất -> đó cũng là góc cần xoay ảnh gốc
    return best_a


def find_document_quad(img: np.ndarray, min_area_ratio: float, max_area_ratio: float) -> np.ndarray | None:
    """Tứ giác bao biên lai (nền sáng trên nền tối hoặc ngược lại) hoặc None nếu không đủ chắc chắn."""
    h, w = img.shape[:2]
    scale = min(1.0, 500 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    g = cv2.GaussianBlur(_gray(small), (7, 7), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    best, best_area = None, 0.0
    for mask in (bw, 255 - bw):                      # biên lai sáng trên nền tối, hoặc ngược lại
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            ratio = area / (small.shape[0] * small.shape[1])
            if area <= best_area or not (min_area_ratio <= ratio <= max_area_ratio):
                continue
            quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(quad) == 4 and cv2.isContourConvex(quad):
                best, best_area = quad.reshape(4, 2).astype(np.float32), area
    if best is None:
        return None
    ratio = best_area / (small.shape[0] * small.shape[1])
    if not (min_area_ratio <= ratio <= max_area_ratio):
        return None
    return best / scale


def preprocess(img: np.ndarray, steps: dict) -> Preprocessed:
    """steps: khoá trong {perspective, deskew, clahe, denoise}, giá trị là tham số (xem configs/ocr.yaml). {} = giữ nguyên ảnh."""
    H = np.eye(3)
    meta: dict = {}
    out = img
    if "perspective" in steps:
        p = steps["perspective"]
        quad = find_document_quad(out, p["min_area_ratio"], p["max_area_ratio"])
        if quad is None:
            meta["perspective"] = "skipped"
        else:
            q = order_quad_pts(quad)
            tw = round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
            th = round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
            dst = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(q, dst)
            out = cv2.warpPerspective(out, M, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            H = M @ H
            meta["perspective"] = "applied"
    if "deskew" in steps:
        p = steps["deskew"]
        a = estimate_skew(out, p["max_angle"])
        meta["skew_deg"] = round(a, 2)
        if abs(a) >= p["min_angle"]:
            R, (nw, nh) = _rotation_homography(a, out.shape[1], out.shape[0])
            out = cv2.warpAffine(out, R[:2], (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            H = R @ H
            meta["deskew"] = "applied"
        else:
            meta["deskew"] = "below_threshold"
    if "clahe" in steps:
        p = steps["clahe"]
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=(p["tile"], p["tile"])).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if "denoise" in steps:
        out = cv2.fastNlMeansDenoisingColored(out, None, steps["denoise"]["h"], steps["denoise"]["h"], 7, 21)
    return Preprocessed(out, H, meta)


def order_quad_pts(pts: np.ndarray) -> np.ndarray:
    return order_quad(pts)
