"""Trích 4 trường MC-OCR (người bán, địa chỉ, ngày/giờ, tổng tiền) từ các dòng OCR bằng luật.

Baseline minh bạch cho luồng đầu tiên; LayoutXLM (M3) thay thế qua cùng đầu ra `dict[str, FieldResult]`.
Độ tin cậy trường = độ tin cậy luật (khớp mạnh/yếu) x rec_score của dòng chứa giá trị, theo công thức token_conf trong kế hoạch mục 5.
Chỉ phụ thuộc thư viện chuẩn + numpy/OpenCV (qua src.ocr.geometry) nên chạy trong `.venv` chung.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from math import hypot
from statistics import median

from src.ocr.geometry import reading_rows
from src.ocr.types import OcrLine
from src.validation.normalize import parse_date, parse_time, parse_vnd

MIN_AMOUNT, MAX_AMOUNT = 1_000, 100_000_000

# Số tiền đứng riêng: không dính chữ/số/dấu phân cách phía trước (loại mã 'HDL201005', ngày '13/08/2020', số âm '-12.000'); cho phép dấu cách làm
# phân cách hàng nghìn ('35 000'). Số liền thiếu ngữ cảnh tổng/tiền tệ phải chia hết cho 100 (loại mã giao dịch, điện thoại, mã vạch).
_AMOUNT = re.compile(r"(?<![\w.,/\-])(\d{1,3}(?:[ \u00a0]*[.,][ \u00a0]*\d{3})+(?:[.,]\d{1,2})?"
                     r"|\d{1,3}(?:[ \u00a0]+\d{3})+(?![ \u00a0]+\d)"
                     r"|[1-9]\d{3,8})(?![\d.,/\-])")

TOTAL_STRONG = ("tong cong", "tong so tien thanh toan", "tong tien thanh toan", "tong thanh toan", "tien thanh toan",
                "khach phai tra", "phai thanh toan", "tong tien phai tra", "phai t.toan", "phai tra")
TOTAL_MID = ("thanh toan", "total", "thanh tien")
TOTAL_WEAK = ("tong tien", "tong", "cong tien hang")
TOTAL_HARD_EXCLUDE = ("khach dua", "khach tra", "tien thua", "tra lai", "chiet khau", "khuyen mai", "so luong", "tong sl",
                      "diem", "tem", "phuong thuc", "hinh thuc", "tong ket", "phieu thanh toan", "da thanh toan")
TOTAL_SOFT_EXCLUDE = ("vat", "thue", "tien mat")  # chỉ loại khi không có từ khoá mạnh: 'Tổng cộng (đã gồm VAT)' vẫn là tổng tiền
DISCOUNT = re.compile(r"\bgiam(?:\s+gia\b|\s*[:=]|\s+\d)|\b(?:da|a)\s+giam\b")  # 'giám' cũng fold thành giam: cần cấu trúc giảm giá
SELLER_BRAND = ("sieu thi", "cua hang", "cty", "cong ty", "tnhh", "mart", "store", "shop", "coop", "co.op", "bach hoa", "cafe",
                "nha hang", "tap hoa", "minimart", "nha thuoc", "trung tam", "pharmacy", "highlands", "phuc long", "vincommerce", "vinmart", "winmart", "coffee")
SELLER_EXCLUDE = ("tel", "dt:", "dien thoai", "hotline", "dia chi", "dc:", "mst", "ma so thue", "www", "http", "email", "fax",
                  "hoa don", "hoa on", "ban hang", "phieu", "tinh tien", "so hd", "ngay", "khach hang", "thu ngan",
                  "tra lai", "tien tra", "cam on", "tong cong", "thanh tien", "so luong", "ward", "district", "street")
SELLER_DESCRIPTOR = re.compile(r"(?:coffee|coffees|cafe|tea|tiem tra|nha hang|cua hang|sieu thi|cong ty(?:\s+tnhh)?)"
                               r"(?:\s*(?:&|and|va)\s*(?:coffee|tea|cafe))?", re.I)
SELLER_TIME_LABEL = re.compile(r"\bgio\s*(?:[:=]|\d)|\bgio\s+(?:lam viec|mo cua)\b")  # 'giò' và 'giờ' cùng fold: cần cấu trúc nhãn giờ
ADDRESS_TOKENS = ("dia chi", "duong", "phuong", "quan", "huyen", "tinh", "thanh pho", "ha noi", "ho chi minh", "tphcm", "tp.", "tp ", "p.",
                  "q.", "xa ", "thon", "khu ", "kp.", "ngo ", "hem ", "pho ", "cho ")
ADDRESS_SHORT = re.compile(r"\b(?:tp|p|q|kp)\.\s*(?=\w)|\b(?:p|q)\s*\d+\b|\bso\s+\d+\b")
STREET_NUMBER = re.compile(r"^\s*\d{1,4}(?:[/\-]\d{1,4}[a-z]?)?\s+[^\W\d_]{2,}\b")  # số nhà + tên đường, không cần chữ 'Số'
CONTACT = re.compile(r"\b(tel|dt|sdt|hotline|mst|fax|email|website|www|http)\b|ma so thue|dien thoai")
ADDRESS_KEY_ONLY = re.compile(r"\s*(dia chi|dc|d/c)\s*[:.\-]*\s*")  # chỉ có nhãn 'Địa chỉ:'; giá trị nằm ở dòng kế
ADDRESS_LABEL = re.compile(r"^\s*(?:dia chi|dc|d/c)\s*[:.\-]\s*\S")
DATE_KEYS = ("ngay", "date", "thoi gian")
TIME_SKIP = ("lam viec", "mo cua", "hang ngay", "hotline")


def fold(text: str) -> str:
    """Chữ thường, bỏ dấu (đ -> d): so khớp từ khoá không phụ thuộc OCR đọc đúng dấu hay không."""
    s = unicodedata.normalize("NFD", (text or "").replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn")).casefold()


def _has_keyword(key: str, keywords: tuple[str, ...]) -> bool:
    """Khớp nguyên từ/cụm; dấu câu trong từ khoá được giữ, không bắt giữa Hotel/Intel/item."""
    return any(re.search(r"\b" + re.escape(word.strip()) + r"(?!\w)", key) for word in keywords)


@dataclass
class FieldResult:
    value: str | int | None
    confidence: float
    rule: str
    evidence: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)   # tín hiệu thô cho hiệu chỉnh độ tin cậy (src/pipelines/calibration.py)


def _score(line: OcrLine) -> float:
    return float(line.rec_score) if line.rec_score is not None else 0.5


def _cx(line: OcrLine) -> float:
    return sum(p[0] for p in line.polygon) / len(line.polygon)


def amounts_in(text: str) -> list[int]:
    key = fold(text)
    if CONTACT.search(key):
        return []
    out = []
    for m in _AMOUNT.finditer(text or ""):
        value = parse_vnd(m[1])
        # Chỉ nới điều kiện số tròn khi số đứng ngay sau nhãn tổng hoặc sát ký hiệu tiền.
        prefix, suffix = fold(text[:m.start()]), fold(text[m.end():])
        total_context = any(re.search(r"\b" + re.escape(k) + r"(?!\w)\s*[:=]?\s*$", prefix)
                            for k in (*TOTAL_STRONG, *TOTAL_MID, *TOTAL_WEAK)) and not _has_keyword(key, TOTAL_HARD_EXCLUDE)
        currency = re.search(r"\b(?:vnd|d)\s*[:=]?\s*$", prefix) or re.match(r"\s*(?:vnd|d)(?!\w)", suffix)
        if value is None or not MIN_AMOUNT <= value <= MAX_AMOUNT or (m[1].isdigit() and value % 100 and not (total_context or currency)):
            continue
        out.append(value)
    return out


def _total(rows: list[list[OcrLine]]) -> FieldResult:
    """Dòng chứa từ khoá 'tổng...' + số tiền: cùng dòng, cùng hàng bên phải, hoặc hàng liền dưới."""
    cands = []  # (key_weight, amount, value_line, key_line, where)
    every = [(a, ln) for row in rows for ln in row for a in amounts_in(ln.text)]
    top = max((a for a, _ in every), default=0)
    for r, row in enumerate(rows):
        for line in row:
            key = fold(line.text)
            weight = (3 if _has_keyword(key, TOTAL_STRONG) else 2 if _has_keyword(key, TOTAL_MID)
                      else 1 if _has_keyword(key, TOTAL_WEAK) else 0)
            if not weight or _has_keyword(key, TOTAL_HARD_EXCLUDE) or DISCOUNT.search(key) or (weight < 3 and _has_keyword(key, TOTAL_SOFT_EXCLUDE)):
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
        feats = {"ocr_conf": _score(vline), "agree": counts[amount], "key_weight": weight, "is_max": float(amount == top), "n_cands": len(cands),
                 "distinct": len(counts)}
        return FieldResult(amount, min(rule_conf, 1.0) * _score(vline), f"keyword:{where}", [kline.text, vline.text], feats)
    if every:
        amount, vline = max(every, key=lambda t: t[0])
        return FieldResult(amount, 0.4 * _score(vline), "fallback:max_amount", [vline.text],
                           {"ocr_conf": _score(vline), "agree": 0, "key_weight": 0, "is_max": 1.0, "n_cands": 0, "distinct": 0})
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
    feats = {"ocr_conf": _score(line), "keyword": float(weight == 2), "n_distinct": len({c[3] for c in cands}), "agree": sum(c[3] == iso for c in cands)}
    return FieldResult(iso, (0.95 if weight == 2 else 0.8) * _score(line), "date_regex", [line.text], feats)


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
            return FieldResult(t, (0.9 if rank == 0 else 0.75) * _score(line), "time_regex", [line.text],
                               {"ocr_conf": _score(line), "same_line": float(rank == 0)})
    return FieldResult(None, 0.0, "none")


def _bounds(line: OcrLine) -> tuple[float, float, float, float]:
    xs, ys = zip(*line.polygon, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def _text_height(line: OcrLine) -> float:
    """Cạnh ngắn trung bình của box: box xoay dọc không được coi là font chữ rất lớn."""
    p = line.polygon
    edges = [hypot(p[(i + 1) % len(p)][0] - x, p[(i + 1) % len(p)][1] - y) for i, (x, y) in enumerate(p)]
    return max(min((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2), 1)


def _seller_brand(key: str) -> bool:
    """Từ thương hiệu nguyên vẹn; các tên ghép mart/co.op phải nằm trong cùng một token."""
    return _has_keyword(key, SELLER_BRAND) or bool(re.search(r"\b(?:co\.?op\w*|\w+mart)\b", key))


def _seller_text(line: OcrLine, nearby: list[OcrLine]) -> str:
    """Cắt phần địa chỉ có cấu trúc; chỉ bỏ hậu tố chi nhánh khi nó lặp trong dòng địa chỉ gần đó."""
    text = line.text.strip()
    for sep in re.finditer(r"\s+[-–—|]\s+|\b(?:địa chỉ|dia chi|đc|dc)\s*:", text, re.I):
        suffix = OcrLine(line.polygon, text=text[sep.end():])
        if _address_hits(suffix) >= 2:
            text = text[:sep.start()].strip()
            break
    words = list(re.finditer(r"\S+", text))
    for i in range(1, len(words) - 1):
        prefix, suffix = text[:words[i].start()].strip(), text[words[i].start():]
        if SELLER_DESCRIPTOR.fullmatch(fold(prefix)) or not _seller_brand(fold(prefix)):
            continue
        if any(ln is not line and _address_hits(ln) >= 2 and _has_keyword(fold(ln.text), (fold(suffix),)) for ln in nearby):
            return prefix
    return text


def _title_neighbors(a: OcrLine, b: OcrLine, a_text: str, b_text: str, body_height: float) -> bool:
    """Hai mảnh tiêu đề phải gần nhau, cỡ chữ tương tự và có dấu hiệu tiêu đề/tên loại cửa hàng."""
    ax0, ay0, ax1, ay1 = _bounds(a)
    bx0, by0, bx1, by1 = _bounds(b)
    ah, bh = max(ay1 - ay0, 1), max(by1 - by0, 1)
    if min(ah, bh) / max(ah, bh) < 0.6:
        return False
    same_row = abs((ay0 + ay1 - by0 - by1) / 2) <= 0.5 * max(ah, bh)
    if same_row:
        close = max(ax0, bx0) - min(ax1, bx1) <= 2 * max(ah, bh)
    else:
        close = (max(ay0, by0) - min(ay1, by1) <= 1.5 * max(ah, bh)
                 and abs((ax0 + ax1 - bx0 - bx1) / 2) <= 0.5 * max(ax1 - ax0, bx1 - bx0))
    descriptor = any(SELLER_DESCRIPTOR.fullmatch(fold(t)) for t in (a_text, b_text))
    prominent = min(ah, bh) >= 1.3 * body_height and a_text.isupper() and b_text.isupper()
    return close and (descriptor or prominent)


def _seller(rows: list[list[OcrLine]]) -> tuple[FieldResult, int]:
    """Chọn tiêu đề trong 8 dòng đầu bằng thương hiệu/cỡ chữ; gộp tối đa 3 mảnh tiêu đề gần nhau."""
    flat = [ln for row in rows for ln in row]
    body_height = median(_text_height(ln) for ln in flat[:12]) if flat else 1
    candidates = {}
    best_i, best_score = -1, 0.0
    for i, line in enumerate(flat[:8]):
        text = _seller_text(line, flat[:12])
        key = fold(text)
        letters = sum(c.isalpha() for c in text)
        if letters < 3 or letters / max(len(text), 1) < 0.6 or _has_keyword(key, SELLER_EXCLUDE) or SELLER_TIME_LABEL.search(key):
            continue
        brand = _seller_brand(key)
        if not brand and _address_hits(OcrLine(line.polygon, text=text)) >= 2:
            continue
        upper = sum(c.isupper() for c in text) / letters
        height = _text_height(line)
        score = 3 * brand + (upper > 0.8) + _score(line) - 0.15 * i + 2 * min(max(height / body_height - 1, 0), 2)
        candidates[i] = text
        if score > best_score:
            best_i, best_score = i, score
    if best_i < 0:
        return FieldResult(None, 0.0, "none"), -1
    chosen = [best_i]
    for direction in (-1, 1):
        while len(chosen) < 3:
            edge = min(chosen) if direction < 0 else max(chosen)
            i = edge + direction
            if i not in candidates or not _title_neighbors(flat[edge], flat[i], candidates[edge], candidates[i], body_height):
                break
            if fold(candidates[i]) == fold(candidates[edge]):
                break  # logo lặp: không nhân đôi tên
            chosen.append(i)
    chosen.sort()
    text = " ".join(candidates[i] for i in chosen)
    brand = _seller_brand(fold(text))
    ocr_conf = min(_score(flat[i]) for i in chosen)
    feats = {"ocr_conf": ocr_conf, "brand": float(brand), "index": chosen[0], "n_lines": len(chosen)}
    return FieldResult(text, (0.9 if brand else 0.65) * ocr_conf, "top_lines", [flat[i].text for i in chosen], feats), chosen[0]


def _address_hits(line: OcrLine) -> int:
    key = fold(line.text)
    if CONTACT.search(key) or ADDRESS_KEY_ONLY.fullmatch(key) or _has_keyword(key, ("hoa don", "so hd", "so ca", "so luong", "ngay", "tong", "doanh so")):
        return 0
    hits = (sum(_has_keyword(key, (tok,)) for tok in ADDRESS_TOKENS if not tok.endswith('.'))
            + len(ADDRESS_SHORT.findall(key)) + 2 * bool(STREET_NUMBER.search(key)) + 2 * bool(ADDRESS_LABEL.search(key)))
    return hits + (2 if _has_keyword(key, ("dia chi",)) else 0) + (1 if hits and re.search(r"\d", key) else 0)


def _address(rows: list[list[OcrLine]], seller_i: int, window: int = 12, span: int = 3, near: int = 4,
             seller_lines: list[str] | None = None) -> FieldResult:
    """Địa chỉ thường trải trên 1-3 dòng và có thể nằm trước hoặc sau tên người bán.

    Lấy dòng nhiều từ khoá địa chỉ nhất trong `window` dòng đầu (hoà thì gần tên người bán hơn), rồi nối các dòng địa chỉ khác
    cách nó không quá `near` dòng, theo thứ tự đọc.
    """
    flat = [ln for row in rows for ln in row][:window]
    hits = {i: _address_hits(ln) for i, ln in enumerate(flat) if i != seller_i and ln.text not in (seller_lines or [])}
    # Nhãn tách riêng là bằng chứng cấu trúc: dòng kế có thể chỉ là tên đường, không có token 'đường/phường'.
    for i, ln in enumerate(flat[:-1]):
        if not ADDRESS_KEY_ONLY.fullmatch(fold(ln.text)) or i + 1 not in hits:
            continue
        value = flat[i + 1]
        if (sum(c.isalpha() for c in value.text) >= 3 and not CONTACT.search(fold(value.text))
                and not _has_keyword(fold(value.text), SELLER_EXCLUDE) and not parse_date(value.text) and not amounts_in(value.text)):
            hits[i + 1] = max(hits[i + 1], 2)
    # dòng liền dưới tên người bán chỉ cần 1 từ khoá ('Chợ Sủi Phú Thị Gia Lâm'); các dòng khác cần >= 2
    ranked = sorted((i for i, h in hits.items() if h >= 2 or (h >= 1 and i == seller_i + 1 >= 1)), key=lambda i: (-hits[i], abs(i - seller_i) if seller_i >= 0 else i))
    if not ranked:
        return FieldResult(None, 0.0, "none")
    head = ranked[0]
    chosen = sorted([head, *[i for i in ranked[1:] if abs(i - head) <= near][: span - 1]])
    prefix = re.compile(r"^\s*(địa chỉ|dia chi|đc|dc|d/c)(?!\w)\s*[:.\-]?\s*", re.I)
    parts = [prefix.sub("", flat[i].text).strip() for i in chosen]
    parts = [x for x in parts if x]
    if not parts:
        return FieldResult(None, 0.0, "none")
    conf = min(_score(flat[i]) for i in chosen) * (0.9 if hits[head] >= 3 else 0.7)
    feats = {"ocr_conf": min(_score(flat[i]) for i in chosen), "hits": hits[head], "n_lines": len(chosen)}
    return FieldResult(" ".join(parts), conf, "address_tokens", [flat[i].text for i in chosen], feats)


def extract_fields(lines: list[OcrLine], today: date | None = None) -> dict[str, FieldResult]:
    """-> {merchant_name, merchant_address, date, time, total}; trường không tìm thấy có value=None, confidence=0."""
    rows = reading_rows(lines)
    seller, seller_i = _seller(rows)
    date_f = _dates(rows, today or date.today())
    return {"merchant_name": seller, "merchant_address": _address(rows, seller_i, seller_lines=seller.evidence), "date": date_f,
            "time": _time(rows, date_f.evidence[0] if date_f.evidence else None), "total": _total(rows)}
