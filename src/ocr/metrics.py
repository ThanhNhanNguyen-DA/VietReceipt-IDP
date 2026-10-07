"""CER/WER theo kế hoạch mục 4.1: chuẩn hoá Unicode NFC trước khi so (dấu tiếng Việt có hai cách mã hoá).

Tổng hợp ở mức corpus: CER = tổng số phép sửa / tổng độ dài tham chiếu (không trung bình CER từng mẫu, vì mẫu ngắn sẽ chi phối).
Ba biến thể để phân tích lỗi: nguyên văn, không phân biệt hoa/thường, không dấu (lỗi dấu tách khỏi lỗi chữ cái).
"""
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s or "")).strip()


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


VARIANTS = {
    "exact": lambda s: s,
    "nocase": lambda s: s.casefold(),
    "nodiac": lambda s: strip_diacritics(s).casefold(),
}


@dataclass(frozen=True)
class Edit:
    """Số phép sửa và độ dài tham chiếu của một cặp (ref, hyp); cộng dồn được."""
    errors: int
    ref_len: int


def char_edit(ref: str, hyp: str, variant: str = "exact") -> Edit:
    f = VARIANTS[variant]
    r, h = f(norm(ref)), f(norm(hyp))
    return Edit(Levenshtein.distance(r, h), len(r))


def word_edit(ref: str, hyp: str, variant: str = "exact") -> Edit:
    f = VARIANTS[variant]
    r, h = f(norm(ref)).split(), f(norm(hyp)).split()
    return Edit(Levenshtein.distance(r, h), len(r))


def rate(edits: list[Edit]) -> float:
    n = sum(e.ref_len for e in edits)
    return sum(e.errors for e in edits) / n if n else float("nan")


def score_pair(ref: str, hyp: str) -> dict[str, int]:
    """Các đếm thô của một cặp; dùng để gộp corpus và bootstrap theo nhóm."""
    out: dict[str, int] = {}
    for v in VARIANTS:
        c, w = char_edit(ref, hyp, v), word_edit(ref, hyp, v)
        out[f"char_err_{v}"], out[f"char_len_{v}"] = c.errors, c.ref_len
        out[f"word_err_{v}"], out[f"word_len_{v}"] = w.errors, w.ref_len
    out["exact_match"] = int(norm(ref) == norm(hyp))
    return out
