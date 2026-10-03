import pytest

from src.validation.normalize import parse_date, parse_time, parse_vnd


@pytest.mark.parametrize("text,expected", [
    ("1.234.567", 1234567), ("1,234,567đ", 1234567), ("16,200.00", 16200), ("150K", 150000),
    ("100.000 Đ", 100000), ("74,000", 74000), ("Tong so tien thanh toan: 36,490.00", 36490),
    ("20.900", 20900), ("1.234,50", 1235), ("475000", 475000), ("Tổng tiền:", None), ("", None),
])
def test_parse_vnd(text, expected):
    assert parse_vnd(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Ngày : 11/08/2020 08:06", "2020-08-11"), ("01-10-26", "2026-10-01"), ("14.08.2020", "2020-08-14"),
    ("Ngày 01 tháng 10 năm 2026", "2026-10-01"), ("31/02/2020", None), ("Thời gian: 15:16:02 - 14/08/2020", "2020-08-14"), ("THỜI GIAN", None),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Ngày : 11/08/2020 08:06", "08:06:00"), ("20 : 42 : 52", "20:42:52"), ("Thời gian: 17:39", "17:39:00"),
    ("NGÀY:", None),
])
def test_parse_time(text, expected):
    assert parse_time(text) == expected
