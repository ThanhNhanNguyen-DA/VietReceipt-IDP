"""Hình học cho OCR: crop dòng chữ từ polygon, ghép dòng dự đoán vào vùng GT, thứ tự đọc. Chỉ cần numpy + OpenCV."""
import cv2
import numpy as np

from .types import OcrLine, Polygon


def order_quad(pts: np.ndarray) -> np.ndarray:
    """4 điểm -> thứ tự trái-trên, phải-trên, phải-dưới, trái-dưới."""
    pts = np.asarray(pts, dtype=np.float32)
    center = pts.mean(axis=0)
    angle = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angle)]
    # Selecting min(sum)/min(diff) independently duplicates corners at 45 degrees.
    # Order all four once, then rotate to TL with a deterministic tie break.
    start = np.lexsort((ordered[:, 0], ordered[:, 1], ordered.sum(axis=1)))[0]
    return np.roll(ordered, -start, axis=0)


def crop_polygon(img: np.ndarray, polygon: Polygon, vertical_ratio: float = 1.5, pad_px: int = 2) -> np.ndarray | None:
    """Cắt và nắn thẳng một dòng chữ (minAreaRect + perspective). Crop quá dọc thì xoay 90 độ như PaddleOCR."""
    rect = cv2.minAreaRect(np.asarray(polygon, dtype=np.float32))
    (cx, cy), (w, h), ang = rect
    if min(w, h) < 2:
        return None
    box = order_quad(cv2.boxPoints(((cx, cy), (w + 2 * pad_px, h + 2 * pad_px), ang)))
    tw = round(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    th = round(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if tw < 2 or th < 2:
        return None
    dst = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box, dst), (tw, th), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if th / tw >= vertical_ratio:
        out = np.rot90(out)
    return np.ascontiguousarray(out)


def apply_homography(polygon: Polygon, H: np.ndarray) -> Polygon:
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2).tolist()


def polygon_area(poly) -> float:
    return float(abs(cv2.contourArea(np.asarray(poly, dtype=np.float32))))


def overlap_ratio(line_poly: Polygon, rect_xywh: list[float]) -> float:
    """Phần diện tích của dòng dự đoán nằm trong hình chữ nhật GT (0..1)."""
    pts = cv2.convexHull(np.asarray(line_poly, dtype=np.float32))
    area = polygon_area(pts)
    if area <= 0:
        return 0.0
    x, y, w, h = rect_xywh
    rect = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(pts, rect)
    return float(inter) / area


def reading_rows(lines: list[OcrLine], y_tol: float = 0.5, H: np.ndarray | None = None) -> list[list[OcrLine]]:
    """Gom dòng thành các hàng (trên xuống dưới), mỗi hàng sắp trái sang phải trong frame H (nếu có); giữ nguyên polygon đầu ra.

    Hai dòng cùng hàng nếu tâm y lệch < y_tol x chiều cao dòng trung vị.
    """
    if not lines:
        return []
    info = []
    for ln in lines:
        p = np.asarray(apply_homography(ln.polygon, H) if H is not None else ln.polygon, dtype=np.float32)
        info.append((ln, float(p[:, 1].mean()), float(p[:, 0].mean()), float(np.ptp(p[:, 1]))))
    tol = y_tol * max(float(np.median([i[3] for i in info])), 1.0)
    rows: list[list[tuple]] = []
    for item in sorted(info, key=lambda i: i[1]):
        if rows and abs(item[1] - np.mean([r[1] for r in rows[-1]])) <= tol:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [[it[0] for it in sorted(row, key=lambda i: i[2])] for row in rows]


def reading_order(lines: list[OcrLine], y_tol: float = 0.5, H: np.ndarray | None = None) -> list[OcrLine]:
    """Trên xuống dưới, trái sang phải (xem reading_rows)."""
    return [ln for row in reading_rows(lines, y_tol, H) for ln in row]


def assign_lines_to_regions(lines: list[OcrLine], regions: list[dict], min_overlap: float = 0.5) -> dict[int, list[OcrLine]]:
    """Gán mỗi dòng dự đoán cho vùng GT chứa nó nhiều nhất (>= min_overlap). regions[i]['bbox'] = [x, y, w, h]. -> {chỉ số vùng: [dòng]}."""
    out: dict[int, list[OcrLine]] = {i: [] for i in range(len(regions))}
    for ln in lines:
        best, best_r = -1, min_overlap
        for i, reg in enumerate(regions):
            r = overlap_ratio(ln.polygon, reg["bbox"])
            if r >= best_r:
                best, best_r = i, r
        if best >= 0:
            out[best].append(ln)
    return out
