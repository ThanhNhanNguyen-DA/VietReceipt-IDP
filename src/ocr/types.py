"""Kiểu dữ liệu OCR dùng chung giữa các môi trường (.venv-ocr / .venv-kie): chỉ phụ thuộc thư viện chuẩn."""
from dataclasses import asdict, dataclass, field

Polygon = list[list[float]]   # [[x, y], ...] theo thứ tự quanh box, toạ độ pixel


@dataclass
class OcrLine:
    polygon: Polygon                       # toạ độ trong ảnh GỐC (đã map ngược qua homography tiền xử lý)
    det_score: float | None = None
    text: str = ""
    rec_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageResult:
    image_id: str
    preproc: str
    lines: list[OcrLine] = field(default_factory=list)
    timing_s: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"image_id": self.image_id, "preproc": self.preproc, "lines": [ln.to_dict() for ln in self.lines],
                "timing_s": self.timing_s, "meta": self.meta}
