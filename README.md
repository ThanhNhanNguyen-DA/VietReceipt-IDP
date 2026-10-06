# VietReceipt IDP

Pipeline Intelligent Document Processing cho biên lai bán lẻ tiếng Việt: tiền xử lý ảnh, OCR (PaddleOCR), trích xuất trường và dòng hàng (LayoutXLM), kiểm tra luật nghiệp vụ, hàng đợi review, API + Supabase.
Kế hoạch đầy đủ: `ai_completed_plan.docx`. Dự án phi thương mại (LayoutXLM/LayoutLMv3 theo giấy phép CC BY-NC-SA 4.0).

## Bắt đầu nhanh (WSL2/Linux, Python 3.11)

```bash
make setup          # tạo .venv, cài requirements-dev.txt, tạo .env từ .env.example
# điền SUPABASE_* trong .env (DB URL dùng Session pooler, xem chú thích trong .env.example)
make test && make lint
make hooks          # (tuỳ chọn) pre-commit
make help           # danh sách lệnh còn lại
```

Môi trường tách theo mục đích (Paddle và PyTorch xung đột CUDA): `.venv` (chung/dev), `.venv-ls` (Label Studio), `.venv-ocr` và `.venv-kie` (sẽ tạo ở M2/M3).

## Cấu hình

| File | Nội dung |
|---|---|
| `.env` (từ `.env.example`) | Secret và URL theo môi trường, đọc bởi `src/config.py` (`Settings`) |
| `configs/pipeline.yaml` | Queue, luật validation, ngưỡng routing, bootstrap CI (`PipelineConfig`, kiểm tra kiểu) |
| `configs/data.yaml` | Dữ liệu huấn luyện MC-OCR, augmentation, mặc định train cho GPU 6 GB |
| `configs/mlflow.yaml` | 5 experiment MLflow (`make mlflow-init`) |
| `configs/receipt_schema.json` | JSON Schema của payload (sinh bằng `make schema`) |
| `configs/label_studio/receipt_kie.xml` | Giao diện gán nhãn |
| `pyproject.toml` | pytest, ruff, coverage |

Dữ liệu được version bằng DVC (`dvc pull`); ảnh và `.env` không bao giờ commit. Hướng dẫn gán nhãn: `docs/ANNOTATION_GUIDELINE.md`, nguồn dữ liệu: `docs/DATASETS.md`.
