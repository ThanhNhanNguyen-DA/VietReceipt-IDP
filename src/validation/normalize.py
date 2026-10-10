"""Chuẩn hoá số tiền VND và ngày giờ trên biên lai (dayfirst)."""
import re
import unicodedata
from datetime import date, time
from decimal import ROUND_HALF_UP, Decimal

_NUM = re.compile(r"\d[\d.,]*")
_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?!\d)")
_DATE_VN = re.compile(r"ng[àa]y\s*(\d{1,2})\s*th[áa]ng\s*(\d{1,2})\s*n[ăa]m\s*(\d{4})", re.I)
_SPACE_GROUPED = re.compile(r"(?<![\d.,:/\-])(\d{1,3})((?:[ \u00a0]\d{3})+)(?![\d:/])")
_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s*[:h]\s*([0-5]\d)(?:\s*[:]\s*([0-5]\d))?(?!\d)")


def parse_vnd(text: str) -> int | None:
    """'1.234.567' / '1,234,567đ' / '16,200.00' / '150K' / '100.000 Đ' -> số nguyên VND. None nếu không có số."""
    s = unicodedata.normalize("NFC", text or "")
    s = _SPACE_GROUPED.sub(lambda m: m[1] + re.sub(r"\s", "", m[2]), s)  # '64 000' -> '64000'
    nums = _NUM.findall(s)
    if not nums:
        return None
    tok = nums[-1].rstrip(".,")
    mult = 1000 if re.search(re.escape(tok) + r"\s*[kK]\b", s) else 1
    seps = [c for c in tok if c in ".,"]
    if not seps:
        return int(tok) * mult
    last = max(tok.rfind("."), tok.rfind(","))
    frac = tok[last + 1:]
    if len(set(seps)) == 2 or (len(seps) == 1 and len(frac) != 3):
        # dấu cuối là phần thập phân (16,200.00 / 1.234,5 / 12.5)
        whole = re.sub(r"[.,]", "", tok[:last])
        return int(Decimal(f"{whole}.{frac}").quantize(Decimal(1), ROUND_HALF_UP)) * mult if whole else None
    return int(re.sub(r"[.,]", "", tok)) * mult  # chỉ có dấu phân cách hàng nghìn


def parse_date(text: str) -> str | None:
    """-> 'YYYY-MM-DD' (đọc ngày trước tháng). None nếu không parse được / ngày không hợp lệ."""
    s = text or ""
    for rx in (_DATE_VN, _DATE):
        pos = 0
        while m := rx.search(s, pos):  # quét chồng lấn: "02 - 14/08/2020" thử "02-14/08" rồi mới tới "14/08/2020"
            d, mo, y = int(m[1]), int(m[2]), int(m[3])
            y += 2000 if y < 100 else 0
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                pos = m.start() + 1
    return None


def parse_time(text: str) -> str | None:
    """-> 'HH:MM:SS'. None nếu không có giờ."""
    m = _TIME.search(text or "")
    return time(int(m[1]), int(m[2]), int(m[3] or 0)).isoformat() if m else None
