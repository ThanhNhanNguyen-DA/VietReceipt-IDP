from datetime import date

from src.extraction.rules import amounts_in, extract_fields, fold
from src.ocr.types import OcrLine

TODAY = date(2026, 10, 10)


def line(text, x, y, score=0.9, w=200, h=24):
    return OcrLine(polygon=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], det_score=0.9, text=text, rec_score=score)


def receipt():
    return [
        line("SIEU THI BACH HOA TONG HOP", 230, 120),
        line("Số 5 Cẩm Tây - Cẩm Phả - QN", 280, 160),
        line("Tel: 0203.3731199", 190, 200),
        line("HÓA ĐƠN BÁN HÀNG", 290, 270),
        line("Ngày: 14-08-2020 21:31:05", 200, 320),
        line("Tổng số:", 185, 880), line("7.00", 314, 880),
        line("Tổng cộng (đã gồm VAT):", 187, 920), line("72,000", 626, 912),
        line("Tiền khách trả:", 194, 960), line("100,000", 627, 952),
    ]


def test_fold():
    assert fold("Tổng CỘNG Đồng") == "tong cong dong"


def test_amounts_filters_quantities_phones_and_barcodes():
    assert amounts_in("72,000") == [72000]
    assert amounts_in("16,200.00") == [16200]
    assert amounts_in("7.00 0945984733 8934673434508") == []


def test_extracts_all_fields():
    f = extract_fields(receipt(), TODAY)
    assert f["merchant_name"].value == "SIEU THI BACH HOA TONG HOP"
    assert f["merchant_address"].value == "Số 5 Cẩm Tây - Cẩm Phả - QN"
    assert f["date"].value == "2020-08-14"
    assert f["time"].value == "21:31:05"
    assert f["total"].value == 72000          # không lấy 'Tiền khách trả' (100,000) dù lớn hơn
    assert f["total"].rule == "keyword:same_row"
    assert all(0 < r.confidence <= 1 for r in f.values())


def test_total_on_next_row_when_key_has_no_amount():
    f = extract_fields([line("Khách phải trả", 70, 1500), line("585,500", 750, 1540)], TODAY)
    assert f["total"].value == 585500 and f["total"].rule == "keyword:next_row"


def test_total_falls_back_to_max_amount_with_low_confidence():
    f = extract_fields([line("Coca", 100, 100), line("12,000", 400, 100), line("35,000", 400, 160)], TODAY)
    assert f["total"].value == 35000 and f["total"].rule == "fallback:max_amount" and f["total"].confidence < 0.4


def test_future_and_ancient_dates_rejected():
    f = extract_fields([line("Ngày: 01/01/2030", 100, 100), line("Ngày: 01/01/1990", 100, 150)], TODAY)
    assert f["date"].value is None and f["date"].confidence == 0


def test_opening_hours_not_taken_as_time():
    f = extract_fields([line("Giờ làm việc: 7:30 đến 22:00", 100, 100), line("Ngày 14/08/2020", 100, 300)], TODAY)
    assert f["time"].value is None


def test_missing_everything_gives_empty_fields():
    f = extract_fields([], TODAY)
    assert all(r.value is None and r.confidence == 0 for r in f.values())
