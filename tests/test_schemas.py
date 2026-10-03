import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.validation.schemas import ReceiptDraft, ReceiptPayload, to_db_rows

VALID = {
    "merchant": {"name": "MINIMART ANAN", "address": "Chợ Sủi, Gia Lâm", "tax_code": "0108123456"},
    "transaction": {"date": "2020-08-11", "time": "08:06:00", "payment_method": "cash"},
    "items": [{"name": "Mì chính 454g", "quantity": "1", "unit_price": "31000", "amount": "31000"},
              {"name": "Thịt heo", "raw_name": "THIT HEO", "quantity": "0.352", "unit_price": "100000", "amount": "35200"}],
    "subtotal": "66200", "total": "66200",
}


def test_valid_payload_roundtrip():
    p = ReceiptPayload.model_validate(VALID)
    assert p.merchant.tax_code == "0108123456" and p.transaction.currency == "VND"
    assert json.loads(p.model_dump_json())["items"][1]["quantity"] == "0.352"


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("total"),
    lambda d: d["merchant"].pop("name"),
    lambda d: d["transaction"].pop("date"),
    lambda d: d.update(items=[]),
    lambda d: d["items"][0].update(amount="-1"),
    lambda d: d["transaction"].update(payment_method="crypto"),
    lambda d: d.update(unknown_field=1),
])
def test_invalid_payload_rejected(mutate):
    d = json.loads(json.dumps(VALID))
    mutate(d)
    with pytest.raises(ValidationError):
        ReceiptPayload.model_validate(d)


def test_draft_allows_missing_fields():
    assert ReceiptDraft.model_validate({}).total is None


def test_to_db_rows_columns_match_migration():
    receipt, items = to_db_rows(ReceiptPayload.model_validate(VALID), "doc-1")
    sql = Path("supabase/migrations/20261002000000_init_schema.sql").read_text(encoding="utf-8")
    body = sql.split("create table public.receipts (")[1].split("\n);")[0]
    cols = {ln.split()[0] for ln in body.splitlines() if ln.strip() and not ln.strip().startswith(("--", "unique", "check"))}
    assert set(receipt) <= cols
    item_body = sql.split("create table public.receipt_items (")[1].split("\n);")[0]
    icols = {ln.split()[0] for ln in item_body.splitlines() if ln.strip() and not ln.strip().startswith(("--", "unique", "check"))}
    assert set(items[0]) <= icols and items[1]["line_no"] == 2


def test_json_schema_file_is_current():
    saved = json.loads(Path("configs/receipt_schema.json").read_text(encoding="utf-8"))
    assert saved == ReceiptPayload.model_json_schema(), "chạy lại: python scripts/export_schema.py"
