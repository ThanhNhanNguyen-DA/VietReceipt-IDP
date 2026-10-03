"""Output schema = hợp đồng giữa Document AI và downstream (kế hoạch mục 2).

- ReceiptDraft:   kết quả trích xuất thô, mọi field tuỳ chọn (có thể thiếu -> đi human review).
- ReceiptPayload: payload đã đủ field bắt buộc; đây là JSON API trả về và là nguồn để ghi DB.
Enum khớp enum Postgres trong supabase/migrations. Tiền là VND (Decimal, >= 0). Luật nghiệp vụ (tổng khớp, VAT, format MST)
nằm ở module validation riêng, không phải ở đây.
"""
import datetime as dt
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[Decimal, Field(ge=0, max_digits=18, decimal_places=2)]
Confidence = Annotated[float, Field(ge=0, le=1)]


class PaymentMethod(StrEnum):
    cash = "cash"
    card = "card"
    ewallet = "ewallet"
    bank_transfer = "bank_transfer"
    other = "other"


class Routing(StrEnum):
    auto_accept = "auto_accept"
    accept_with_warning = "accept_with_warning"
    human_review = "human_review"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------- Draft (tuỳ chọn hết) ----------
class ItemDraft(_Base):
    name: str | None = None
    raw_name: str | None = None
    normalized_name: str | None = None
    quantity: Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=3)] | None = None
    unit_price: Money | None = None
    amount: Money | None = None
    confidence: Confidence | None = None


class MerchantDraft(_Base):
    name: str | None = None
    address: str | None = None
    tax_code: str | None = None        # giữ dạng chuỗi để không mất số 0 đầu


class TransactionDraft(_Base):
    invoice_id: str | None = None
    date: dt.date | None = None
    time: dt.time | None = None
    payment_method: PaymentMethod | None = None
    currency: str = "VND"
    price_includes_vat: bool | None = None


class ReceiptDraft(_Base):
    merchant: MerchantDraft = MerchantDraft()
    transaction: TransactionDraft = TransactionDraft()
    items: list[ItemDraft] = []
    subtotal: Money | None = None
    vat: Money | None = None
    discount: Money | None = None
    service_charge: Money | None = None
    total: Money | None = None
    doc_confidence: Confidence | None = None
    routing: Routing | None = None
    model_version: str | None = None


# ---------- Payload (field bắt buộc theo kế hoạch) ----------
class Item(ItemDraft):
    name: str = Field(min_length=1)
    quantity: Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=3)]
    unit_price: Money
    amount: Money


class Merchant(MerchantDraft):
    name: str = Field(min_length=1)


class Transaction(TransactionDraft):
    date: dt.date


class ReceiptPayload(ReceiptDraft):
    merchant: Merchant
    transaction: Transaction
    items: list[Item] = Field(min_length=1)
    total: Money


def to_db_rows(p: ReceiptPayload, document_id: str, run_id: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """-> (dòng bảng receipts, danh sách dòng receipt_items); cột trùng tên với migration init_schema."""
    receipt = {
        "document_id": document_id, "run_id": run_id,
        "merchant_name": p.merchant.name, "merchant_address": p.merchant.address, "tax_code": p.merchant.tax_code,
        "invoice_id": p.transaction.invoice_id, "transaction_date": p.transaction.date.isoformat(),
        "transaction_time": p.transaction.time.isoformat() if p.transaction.time else None,
        "payment_method": p.transaction.payment_method, "currency": p.transaction.currency,
        "price_includes_vat": p.transaction.price_includes_vat,
        "subtotal": p.subtotal, "vat_amount": p.vat, "discount_amount": p.discount, "service_charge": p.service_charge,
        "total_amount": p.total, "doc_confidence": p.doc_confidence, "routing": p.routing, "model_version": p.model_version,
        "raw_json": p.model_dump(mode="json"),
    }
    items = [{"line_no": i, "raw_name": it.raw_name, "name": it.name, "normalized_name": it.normalized_name,
              "quantity": it.quantity, "unit_price": it.unit_price, "amount": it.amount, "confidence": it.confidence}
             for i, it in enumerate(p.items, start=1)]
    return receipt, items
