"""Trích 4 trường MC-OCR (người bán, địa chỉ, ngày/giờ, tổng tiền) từ các dòng OCR bằng luật.

Baseline minh bạch cho luồng đầu tiên; LayoutXLM (M3) thay thế qua cùng đầu ra `dict[str, FieldResult]`.
Độ tin cậy trường = độ tin cậy luật (khớp mạnh/yếu) x rec_score của dòng chứa giá trị, theo công thức token_conf trong kế hoạch mục 5.
Chỉ phụ thuộc thư viện chuẩn + numpy/OpenCV (qua src.ocr.geometry) nên chạy trong `.venv` chung.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from src.ocr.geometry import reading_rows
from src.ocr.types import OcrLine
from src.validation.normalize import parse_date, parse_time, parse_vnd

MIN_AMOUNT, MAX_AMOUNT = 1_000, 100_000_000

# Số tiền đứng riêng: không dính chữ/số/dấu phân cách phía trước (loại mã 'HDL201005', ngày '13/08/2020', số âm '-12.000'); cho phép dấu cách làm
# phân cách hàng nghìn ('35 000'). Số liền không phân cách phải chia hết cho 100 (loại mã giao dịch, điện thoại, mã vạch).
_AMOUNT = re.compile(r"(?<![\w.,/\-])(\d{1,3}(?:[., \u00a0]\d{3})+(?:[.,]\d{1,2})?|[1-9]\d{3,8})(?![\d/])")

TOTAL_STRONG = ("tong cong", "tong so tien thanh toan", "tong tien thanh toan", "tong thanh toan", "tien thanh toan",
                "khach phai tra", "phai thanh toan", "tong tien phai tra", "phai t.toan", "phai tra")
TOTAL_MID = ("thanh toan", "total", "thanh tien")
TOTAL_WEAK = ("tong tien", "tong", "cong tien hang")
TOTAL_HARD_EXCLUDE = ("khach dua", "khach tra", "tien thua", "tra lai", "chiet khau", "giam", "khuyen mai", "so luong", "tong sl",
                      "diem", "tem", "phuong thuc", "hinh thuc", "tong ket", "phieu thanh toan", "da thanh toan")
TOTAL_SOFT_EXCLUDE = ("vat", "thue", "tien mat")  # chỉ loại khi không có từ khoá mạnh: 'Tổng cộng (đã gồm VAT)' vẫn là tổng tiền
SELLER_BRAND = ("sieu thi", "cua hang", "cty", "cong ty", "tnhh", "mart", "store", "shop", "coop", "co.op", "bach hoa", "cafe",
                "nha hang", "tap hoa", "minimart", "nha thuoc", "trung tam", "pharmacy", "highlands", "phuc long", "vincommerce", "vinmart", "winmart")
SELLER_EXCLUDE = ("tel", "dt:", "dien thoai", "hotline", "dia chi", "dc:", "mst", "ma so thue", "www", "http", "email", "fax",
                  "hoa don", "hoa on", "ban hang", "phieu", "tinh tien", "so hd", "ngay", "gio", "khach hang", "thu ngan")
ADDRESS_TOKENS = ("dia chi", "duong", "phuong", "quan", "huyen", "tinh", "thanh pho", "ha noi", "ho chi minh", "tphcm", "tp.", "tp ", "p.",
                  "q.", "xa ", "thon", "khu ", "kp.", "ngo ", "hem ", "pho ", "so ", "cho ")
CONTACT = re.compile(r"\b(tel|dt|sdt|hotline|mst|fax|email|website|www|http)\b|ma so thue|dien thoai")
ADDRESS_KEY_ONLY = re.compile(r"\s*(dia chi|dc|d/c)\s*[:.\-]*\s*")  # chỉ có nhãn 'Địa chỉ:'; giá trị nằm ở dòng kế
DATE_KEYS = ("ngay", "date", "thoi gian")
TIME_SKIP = ("lam viec", "mo cua", "hang ngay", "hotline")


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu (đ -> d): so khớp từ khoá không phụ thuộc OCR đọc đúng dấu hay không."""
    s = unicodedata.normalize("NFD", (text or "").replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn")).casefold()


@dataclass
class FieldResult:
    value: str | int | None
    confidence: float
    rule: str
    evidence: list[str] = field(default_factory=list)


def _score(line: OcrLine) -> float:
    return float(line.rec_score) if line.rec_score is not None else 0.5


def _cx(line: OcrLine) -> float:
    return sum(p[0] for p in line.polygon) / len(line.polygon)


def amounts_in(text: str) -> list[int]:
    out = []
    for m in _AMOUNT.finditer(text or ""):
        value = parse_vnd(m[1])
        if value is None or not MIN_AMOUNT <= value <= MAX_AMOUNT or (m[1].isdigit() and value % 100):
            continue
        out.append(value)
    return out


def _total(rows: list[list[OcrLine]]) -> FieldResult:
    """Dòng chứa từ khoá 'tổng...' + số tiền: cùng dòng, cùng hàng bên phải, hoặc hàng liền dưới."""
    cands = []  # (key_weight, amount, value_line, key_line, where)
    for r, row in enumerate(rows):
        for line in row:
            key = fold(line.text)
            weight = (3 if any(k in key for k in TOTAL_STRONG) else 2 if any(k in key for k in TOTAL_MID)
                      else 1 if any(k in key for k in TOTAL_WEAK) else 0)
            if not weight or any(x in key for x in TOTAL_HARD_EXCLUDE) or (weight < 3 and any(x in key for x in TOTAL_SOFT_EXCLUDE)):
                continue
            own = amounts_in(line.text)
            if own:
                cands.append((weight, own[-1], line, line, "same_line", r))
                continue
            right = [x for x in row if x is not line and _cx(x) > _cx(line) and amounts_in(x.text)]
            if right:
                v = right[-1]
                cands.append((weight, amounts_in(v.text)[-1], v, line, "same_row", r))
            elif r + 1 < len(rows):
                below = [x for x in rows[r + 1] if amounts_in(x.text)]
                if below:
                    v = below[-1]
                    cands.append((weight - 1, amounts_in(v.text)[-1], v, line, "next_row", r))
    if cands:
        counts: dict[int, int] = {}
        for c in cands:
            counts[c[1]] = counts.get(c[1], 0) + 1
        best = max(cands, key=lambda c: (c[0] + 0.5 * min(counts[c[1]] - 1, 2), c[5]))  # hoà thì lấy dòng thấp nhất trang
        weight, amount, vline, kline, where, _ = best
        rule_conf = {3: 0.95, 2: 0.85, 1: 0.7, 0: 0.6}[max(weight, 0)] + (0.05 if counts[amount] > 1 else 0.0)
        return FieldResult(amount, min(rule_conf, 1.0) * _score(vline), f"keyword:{where}", [kline.text, vline.text])
    every = [(a, ln) for row in rows for ln in row for a in amounts_in(ln.text)]
    if every:
        amount, vline = max(every, key=lambda t: t[0])
        return FieldResult(amount, 0.4 * _score(vline), "fallback:max_amount", [vline.text])
    return FieldResult(None, 0.0, "none")


def _dates(rows: list[list[OcrLine]], today: date) -> FieldResult:
    cands = []  # (weight, row, line, iso)
    for r, row in enumerate(rows):
        for line in row:
            iso = parse_date(line.text)
            if iso and date(2000, 1, 1) <= date.fromisoformat(iso) <= today:
                cands.append((2 if any(k in fold(line.text) for k in DATE_KEYS) else 1, r, line, iso))
    if not cands:
        return FieldResult(None, 0.0, "none")
    weight, _, line, iso = max(cands, key=lambda c: (c[0], -c[1]))  # từ khoá 'ngày' trước, sau đó dòng cao nhất
    return FieldResult(iso, (0.95 if weight == 2 else 0.8) * _score(line), "date_regex", [line.text])


def _time(rows: list[list[OcrLine]], date_line: str | None) -> FieldResult:
    """Giờ ở cùng dòng với ngày, rồi cùng hàng, rồi tối đa 2 hàng dưới; bỏ giờ mở cửa/hotline."""
    anchor = next(((r, ln) for r, row in enumerate(rows) for ln in row if date_line and ln.text == date_line), None)
    if anchor is None:
        order = [(r, ln) for r, row in enumerate(rows) for ln in row]
    else:
        r0, a = anchor
        order = [(r0, a)] + [(r, ln) for r in range(r0, min(r0 + 3, len(rows))) for ln in rows[r] if ln is not a]
    for rank, (_, line) in enumerate(order):
        if any(x in fold(line.text) for x in TIME_SKIP):
            continue
        t = parse_time(line.text)
        if t:
            return FieldResult(t, (0.9 if rank == 0 else 0.75) * _score(line), "time_regex", [line.text])
    return FieldResult(None, 0.0, "none")


def _seller(rows: list[list[OcrLine]]) -> tuple[FieldResult, int]:
    """Dòng giống tên cửa hàng trong 8 dòng đầu: có từ thương hiệu, ít chữ số, không phải dòng liên hệ/tiêu đề."""
    flat = [ln for row in rows for ln in row]
    best, best_i, best_score = None, -1, 0.0
    for i, line in enumerate(flat[:8]):
        text, key = line.text.strip(), fold(line.text)
        letters = sum(c.isalpha() for c in text)
        if letters < 3 or letters / max(len(text), 1) < 0.6 or any(x in key for x in SELLER_EXCLUDE):
            continue
        brand = any(x in key for x in SELLER_BRAND)
        upper = sum(c.isupper() for c in text) / letters
        score = 3 * brand + (upper > 0.8) + _score(line) - 0.15 * i
        if score > best_score:
            best, best_i, best_score = line, i, score
    if best is None:
        return FieldResult(None, 0.0, "none"), -1
    brand = any(x in fold(best.text) for x in SELLER_BRAND)
    return FieldResult(best.text.strip(), (0.9 if brand else 0.65) * _score(best), "top_lines", [best.text]), best_i


def _address_hits(line: OcrLine) -> int:
    key = fold(line.text)
    if CONTACT.search(key) or ADDRESS_KEY_ONLY.fullmatch(key):
        return 0
    return sum(tok in key for tok in ADDRESS_TOKENS) + (2 if "dia chi" in key else 0) + (1 if re.search(r"\d", key) else 0)


def _address(rows: list[list[OcrLine]], seller_i: int, window: int = 12, span: int = 3, near: int = 4) -> FieldResult:
    """Địa chỉ thường trải trên 1-3 dòng và có thể nằm trước hoặc sau tên người bán.

    Lấy dòng nhiều từ khoá địa chỉ nhất trong `window` dòng đầu (hoà thì gần tên người bán hơn), rồi nối các dòng địa chỉ khác
    cách nó không quá `near` dòng, theo thứ tự đọc.
    """
    flat = [ln for row in rows for ln in row][:window]
    hits = {i: _address_hits(ln) for i, ln in enumerate(flat) if i != seller_i}
    # dòng liền dưới tên người bán chỉ cần 1 từ khoá ('Chợ Sủi Phú Thị Gia Lâm'); các dòng khác cần >= 2
    ranked = sorted((i for i, h in hits.items() if h >= 2 or (h >= 1 and i == seller_i + 1 >= 1)), key=lambda i: (-hits[i], abs(i - seller_i) if seller_i >= 0 else i))
    if not ranked:
        return FieldResult(None, 0.0, "none")
    head = ranked[0]
    chosen = sorted([head, *[i for i in ranked[1:] if abs(i - head) <= near][: span - 1]])
    prefix = re.compile(r"^\s*(địa chỉ|dia chi|đc|dc)\s*[:.\-]?\s*", re.I)
    parts = [prefix.sub("", flat[i].text).strip() for i in chosen]
    parts = [x for x in parts if x]
    if not parts:
        return FieldResult(None, 0.0, "none")
    conf = min(_score(flat[i]) for i in chosen) * (0.9 if hits[head] >= 3 else 0.7)
    return FieldResult(" ".join(parts), conf, "address_tokens", [flat[i].text for i in chosen])


def extract_fields(lines: list[OcrLine], today: date | None = None) -> dict[str, FieldResult]:
    """-> {merchant_name, merchant_address, date, time, total}; trường không tìm thấy có value=None, confidence=0."""
    rows = reading_rows(lines)
    seller, seller_i = _seller(rows)
    date_f = _dates(rows, today or date.today())
    return {"merchant_name": seller, "merchant_address": _address(rows, seller_i), "date": date_f,
            "time": _time(rows, date_f.evidence[0] if date_f.evidence else None), "total": _total(rows)}
