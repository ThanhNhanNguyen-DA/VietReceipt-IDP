"""Xuất JSON Schema của ReceiptPayload -> configs/receipt_schema.json (dùng cho contract test / tài liệu API).
Chạy: python scripts/export_schema.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.schemas import ReceiptPayload  # noqa: E402

out = Path("configs/receipt_schema.json")
out.write_text(json.dumps(ReceiptPayload.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("wrote", out)
