# VietReceipt IDP

VietReceipt IDP xử lý biên lai bán lẻ tiếng Việt. Ưu tiên hiện tại là luồng ảnh MC-OCR, dùng **PaddleOCR detection + VietOCR recognition**.
Dự án phi thương mại (LayoutXLM/LayoutLMv3 theo giấy phép CC BY-NC-SA 4.0). Fine-tune đang tạm dừng, chỉ tiếp tục khi có yêu cầu.

## Bắt đầu nhanh (WSL2/Linux, Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp -n .env.example .env
# điền SUPABASE_* trong .env (DB URL dùng Session pooler, xem chú thích trong .env.example)
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Git chỉ giữ một file `requirements.txt` cho phụ thuộc chung. Paddle và PyTorch chạy trong hai môi trường riêng:

```bash
# Detection: ví dụ CPU; dùng wheel Paddle GPU phù hợp máy nếu chạy GPU
python3.11 -m venv .venv-ocr
.venv-ocr/bin/python -m pip install -r requirements.txt paddlepaddle paddleocr==3.7.0

# Recognition: cài Torch/Torchvision phù hợp CUDA của máy
python3.11 -m venv .venv-kie
.venv-kie/bin/python -m pip install torch torchvision
.venv-kie/bin/python -m pip install -r requirements.txt vietocr==0.3.12
```

Label Studio (tuỳ chọn): dùng `.venv-ls` riêng và cài `label-studio==1.23.2` cùng `label-studio-sdk>=1.0`.
Các công cụ khác cài khi cần: `matplotlib` để vẽ báo cáo, `mlflow` để chạy tracking server.

## Thư mục

| Thư mục | Nội dung |
|---|---|
| `src/` | Xử lý ảnh, OCR, chuẩn hoá và validation |
| `scripts/` | Lệnh chuẩn bị dữ liệu, chạy OCR và đánh giá |
| `configs/` | Cấu hình pipeline, model và schema |
| `docs/` | Hướng dẫn và báo cáo |
| `tests/` | Kiểm tra mã nguồn |
| `data/` | Dữ liệu local; Git chỉ giữ metadata |
| `docker/`, `api/`, `supabase/` | Cấu hình hạ tầng và tích hợp |

## Cấu hình

| File | Nội dung |
|---|---|
| `.env` (từ `.env.example`) | Secret và URL theo môi trường, đọc bởi `src/config.py` (`Settings`) |
| `configs/pipeline.yaml` | Queue, luật validation, ngưỡng routing, bootstrap CI (`PipelineConfig`, kiểm tra kiểu) |
| `configs/data.yaml` | Dữ liệu huấn luyện MC-OCR, augmentation, mặc định train cho GPU 6 GB |
| `configs/mlflow.yaml` | 5 experiment MLflow (`python scripts/init_mlflow.py`) |
| `configs/receipt_schema.json` | JSON Schema của payload (`python scripts/export_schema.py`) |
| `configs/label_studio/receipt_kie.xml` | Giao diện gán nhãn |
| `pyproject.toml` | pytest, ruff, coverage |

Ảnh, model, outputs và `.env` được giữ local, không commit. Đặt MC-OCR vào `data/raw/mcocr`, rồi chạy `python scripts/prepare_mcocr.py` để tạo manifest và ảnh xử lý.
Các file `.dvc` còn lại lưu metadata của dữ liệu cũ; cấu hình DVC và công cụ cá nhân được giữ local.
Hướng dẫn gán nhãn: [docs/ANNOTATION_GUIDELINE.md](docs/ANNOTATION_GUIDELINE.md); nguồn dữ liệu: [docs/DATASETS.md](docs/DATASETS.md).

## OCR baseline (M2)

Runner `scripts/ocr_baseline.py` tạo crop dùng chung cho PaddleOCR/VietOCR, nhận dạng trong hai môi trường riêng và đánh giá E1/E2 trên vùng MC-OCR có nhãn.
Cấu hình: `configs/ocr.yaml`. Lệnh chạy, phạm vi metric và đầu ra: [docs/OCR_BASELINE.md](docs/OCR_BASELINE.md).

Đã benchmark A/B với ba variant tiền xử lý trên toàn bộ 113 ảnh validation.
Kết quả tốt nhất hiện tại: VietOCR + `none`, E1 CER 12,42%, E2 CER 20,32%; chưa đạt mục tiêu CER 5%.
Báo cáo chất lượng và bước tiếp theo: [docs/OCR_VALIDATION_RESULTS.md](docs/OCR_VALIDATION_RESULTS.md).

Đã thử thêm hướng đọc 0°/180° trên đủ validation: VietOCR E1 CER 12,24%, E2 CER 20,14%.
Variant thử nghiệm còn bỏ sót ảnh lộn và chưa cải thiện accuracy giá trị nghiệp vụ; giữ `none` làm mặc định.
Chi tiết và lệnh tái chạy: [docs/OCR_ORIENTATION_RESULTS.md](docs/OCR_ORIENTATION_RESULTS.md).
