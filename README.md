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

## OCR baseline (M2)

Runner `scripts/ocr_baseline.py` tạo crop dùng chung cho PaddleOCR/VietOCR, nhận dạng trong hai môi trường riêng và đánh giá E1/E2 trên vùng MC-OCR có nhãn.
Cấu hình: `configs/ocr.yaml`. Lệnh chạy, phạm vi metric và đầu ra: [docs/OCR_BASELINE.md](docs/OCR_BASELINE.md).

Đã benchmark A/B với ba variant tiền xử lý trên toàn bộ 113 ảnh validation.
Kết quả tốt nhất hiện tại: VietOCR + `none`, E1 CER 12,42%, E2 CER 20,32%; chưa đạt mục tiêu CER 5%.
Báo cáo chất lượng và bước tiếp theo: [docs/OCR_VALIDATION_RESULTS.md](docs/OCR_VALIDATION_RESULTS.md).

Đã thử thêm hướng đọc 0°/180° trên đủ validation: VietOCR E1 CER 12,24%, E2 CER 20,14%.
Variant thử nghiệm còn bỏ sót ảnh lộn và chưa cải thiện accuracy giá trị nghiệp vụ; giữ `none` làm mặc định.
Chi tiết và lệnh tái chạy: [docs/OCR_ORIENTATION_RESULTS.md](docs/OCR_ORIENTATION_RESULTS.md).

## Luồng đầu tiên trên MC-OCR

VietOCR là recognizer chính (detector Paddle). `make flow` chạy ảnh MC-OCR -> OCR -> trích trường bằng luật -> chuẩn hoá -> routing -> JSON và chấm theo GT.
Trên validation: tổng tiền 78,7%, ngày 86,4%, giờ 61,5%; tên người bán và địa chỉ CER 31,6% và 33,9%.
Luật và kết quả: [docs/RECEIPT_FLOW.md](docs/RECEIPT_FLOW.md).
