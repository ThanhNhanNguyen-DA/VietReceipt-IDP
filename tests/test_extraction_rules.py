from datetime import date

import pytest

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


@pytest.mark.parametrize("seller,with_date", [("Hotel Continental", True), ("Giò chả Bà Hai", False), ("Intel Store", False)])
def test_a1_seller_keywords_do_not_match_inside_names(seller, with_date):
    lines = [line(seller, 0, 0)] + ([line("Ngày: 14/08/2020", 0, 40)] if with_date else [])
    assert extract_fields(lines, TODAY)["merchant_name"].value == seller


@pytest.mark.parametrize("label", ["Giờ: 21:31", "Giờ mở cửa: 7:30 đến 22:00", "Giờ làm việc: 7:30 đến 22:00"])
def test_a1_time_labels_still_are_not_sellers(label):
    assert extract_fields([line(label, 0, 0), line("Hotel Continental", 0, 40)], TODAY)["merchant_name"].value == "Hotel Continental"


@pytest.mark.parametrize("text,expected", [
    ("Tổng số 5 116,000", [116000]), ("SL 2 14,000", [14000]), ("SL 2 140,000", [140000]),
    ("35 000", [35000]), ("1 234 567", [1234567]), ("5 116,000", [116000]),
])
def test_a3_quantity_is_not_a_thousands_group(text, expected):
    assert amounts_in(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Tổng cộng 45280", [45280]), ("Tổng cộng 45300", [45300]), ("45280đ", [45280]),
    ("45280 VND", [45280]), ("vnđ 45280", [45280]),
    ("PHIẾU THANH TOÁN 100022 - 550301", []), ("100022 - 550301", []),
    ("Hóa đơn: HDL201005-0011", []), ("DT:37307551", []),
])
def test_a4_non_round_amount_needs_total_or_currency_context(text, expected):
    assert amounts_in(text) == expected


def test_a5_quantity_phrase_is_not_an_address():
    lines = [line("SIEU THI AN", 0, 0), line("Mua so luong lon giam gia", 0, 40), line("Cam on quy khach", 0, 80)]
    assert extract_fields(lines, TODAY)["merchant_address"].value is None


@pytest.mark.parametrize("address", ["Số 5 Cẩm Tây", "so 32 Duong A", "P.1, Q.3, TP.HCM", "22 Quang Trung, P10, Gò Vấp",
                                     "23 Nguyễn Thị Định - Trung Hòa - Cầu Giấy - HN", "244/23a Huỳnh Văn Bánh"])
def test_a5_structured_address_tokens_are_kept(address):
    assert extract_fields([line("SIEU THI AN", 0, 0), line(address, 0, 40)], TODAY)["merchant_address"].value == address


@pytest.mark.parametrize("text", ["Tổng cộng giám đốc 50,000", "ITEM TOTAL 50,000", "Total private 50,000"])
def test_a6_total_exclusions_match_words_or_discount_structure(text):
    total = extract_fields([line(text, 0, 0)], TODAY)["total"]
    assert total.value == 50000 and total.rule == "keyword:same_line"


@pytest.mark.parametrize("text", ["Tổng tiền đã giảm 100,000", "Tổng tiền Ã giảm 100,000", "Giảm giá 100,000"])
def test_a6_discount_remains_excluded(text):
    total = extract_fields([line("Tổng cộng 50,000", 0, 0), line(text, 0, 40)], TODAY)["total"]
    assert total.value == 50000


@pytest.mark.parametrize("horizontal", [True, False])
def test_b1_join_seller_name_and_business_descriptor(horizontal):
    name = line("THỨC", 0, 0, w=100, h=40)
    descriptor = line("COFFEE", 110 if horizontal else 0, 0 if horizontal else 48, w=150, h=40)
    address = line("22 Quang Trung, P10, Gò Vấp", 0, 100, h=20)
    seller = extract_fields([name, descriptor, address], TODAY)["merchant_name"]
    assert seller.value == "THỨC COFFEE"
    assert seller.evidence == ["THỨC", "COFFEE"]
    assert extract_fields([name, descriptor, address], TODAY)["merchant_address"].value == address.text


def test_b1_trim_structured_address_suffix_from_seller():
    seller = "CO.OPSMILE - A1 KHU B KTX DAI HOC QUOC GIA"
    assert extract_fields([line(seller, 0, 0)], TODAY)["merchant_name"].value == "CO.OPSMILE"
    # Tên có dấu gạch nhưng phần sau không phải địa chỉ phải giữ nguyên.
    assert extract_fields([line("SHOP - AN BINH", 0, 0)], TODAY)["merchant_name"].value == "SHOP - AN BINH"


def test_b1_trim_branch_suffix_only_with_nearby_address_evidence():
    name = line("Co.OPMart HAU GIANG", 0, 0)
    address = line("188 Hau Giang, P.6, Q.6, TpHCM", 0, 40)
    assert extract_fields([name, address], TODAY)["merchant_name"].value == "Co.OPMart"
    assert extract_fields([name], TODAY)["merchant_name"].value == name.text
    # Cùng luật cho một tên khác, không vá theo thương hiệu hay địa danh.
    assert extract_fields([line("Intel Store BAC NINH", 0, 0), line("Số 5 Bắc Ninh", 0, 40)], TODAY)["merchant_name"].value == "Intel Store"


def test_b1_seller_rejects_payment_and_address_lines():
    lines = [line("Trả lại khách", 0, 0), line("ching ward", 0, 40), line("AN BINH", 0, 80, h=40)]
    assert extract_fields(lines, TODAY)["merchant_name"].value == "AN BINH"


def test_b1_keeps_separate_headers_and_uses_title_height():
    lines = [line("Tiệm trà", 0, 0, h=20), line("AN BINH", 0, 80, h=48), line("Số 5 Đường A", 0, 145, h=20)]
    assert extract_fields(lines, TODAY)["merchant_name"].value == "AN BINH"


def test_b1_vertical_box_is_not_a_large_title_font():
    lines = [line("QUIWJYO QH", 0, 0, w=24, h=224), line("TEST SHOP", 100, 120, w=200, h=40)]
    assert extract_fields(lines, TODAY)["merchant_name"].value == "TEST SHOP"


def test_b4_address_label_anchors_value_without_address_keywords():
    lines = [line("TEST SHOP", 0, 0), line("Địa chỉ:", 0, 40), line("Phan Huy Chú", 0, 80), line("Tel: 028 1234 5678", 0, 120)]
    address = extract_fields(lines, TODAY)["merchant_address"]
    assert address.value == "Phan Huy Chú" and address.evidence == ["Phan Huy Chú"]


@pytest.mark.parametrize("label", ["ĐC:", "DC:", "D/C:"])
def test_b4_address_abbreviation_anchors_inline_street(label):
    lines = [line("TEST SHOP", 0, 0), line(f"{label} 23 Nguyễn Thị Định - Trung Hòa - Cầu Giấy - HN", 0, 40)]
    assert extract_fields(lines, TODAY)["merchant_address"].value == "23 Nguyễn Thị Định - Trung Hòa - Cầu Giấy - HN"


@pytest.mark.parametrize("next_line", ["Tel: 028 1234 5678", "Mã số thuế: 0305781492", "Ngày: 14/08/2020", "50,000"])
def test_b4_address_label_does_not_steal_contact_date_or_amount(next_line):
    lines = [line("TEST SHOP", 0, 0), line("Địa chỉ:", 0, 40), line(next_line, 0, 80)]
    assert extract_fields(lines, TODAY)["merchant_address"].value is None
