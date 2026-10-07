# Rà soát dữ liệu và fine-tune OCR

Hai recognizer được fine-tune riêng trong `.venv-kie` (VietOCR) và `.venv-ocr` (Paddle),
chạy lần lượt trên GPU 6 GB. Detector Paddle được giữ nguyên.
Cấu hình: `configs/ocr_finetune.yaml`; phiên bản dữ liệu: `data/processed/ocr_review_v1.dvc`.

## Dữ liệu và phạm vi review

Audit cấu trúc toàn bộ **5.384 crop** của MC-OCR: 4.770 train, 614 validation, thuộc 832/113 ảnh.
Đối chiếu transcription với CSV nguồn, kiểm tra group split, kích thước, hash, từ điển ký tự và pixel trùng.
Không có group hoặc crop trùng pixel giữa train/val. Không đọc crop test/test_seen.

Kiểm tra trực quan **49 crop**: 24 crop train và 24 validation có CER VietOCR cao nhất,
cộng lỗi số quầy `04` đã phát hiện. Prediction chỉ dùng xếp thứ tự review.
Sửa nhãn bằng pixel/zoom, không tự lấy OCR output làm ground truth.
Các sửa đổi nằm trong [ocr_label_corrections_v1.json](../configs/ocr_label_corrections_v1.json):

| Split | Sửa transcription | Xoay crop 180° | Loại crop | Còn lại |
|---|---:|---:|---:|---:|
| train | 9 | 2 | 6 | 4.764 |
| val | 6 | 5 | 9 | 605 |

Ví dụ: `Ngày` bị gán cho cả dòng ngày/giờ; `Total` bị gán cho cả dòng `Total: 51,000`;
`600đ` bị gán cho crop `GIỜ:`; `04 04` thực tế là `04`.
Loại crop có transcription chỉ bao phủ một phần của nhiều dòng, crop cả trang hoặc chữ không đủ rõ
để sửa chắc chắn. Không loại mẫu chỉ vì OCR đọc sai. Có crop nguồn đã đúng chiều nhưng cũng có crop
lộn 180°; không giả định toàn bộ crop do dataset cung cấp đều đã đúng hướng.

Đây là **audit cấu trúc đầy đủ và review trực quan có chọn lọc**, chưa phải gán nhãn lại thủ công toàn bộ.
Các nhãn chưa review vẫn giữ transcription nguồn. Đề xuất nhãn entity/role cho KIE được ghi riêng,
chưa áp dụng lên manifest KIE hoặc báo cáo benchmark gốc.

`ocr_review_v1` chứa PNG dùng chung, `train/val.jsonl`, `train/val.tsv` và metadata hash.
Mọi trainer kiểm tra hash annotation/crop và leakage trước khi nạp model.
Các sửa đổi chỉ tạo phiên bản mới; dữ liệu raw, `mcocr_clean`, baseline GT và test không bị sửa.

## Tái tạo audit và dữ liệu

Chạy từ gốc repo; mỗi output phải là thư mục mới/rỗng:

```bash
.venv/bin/python scripts/prepare_ocr_training.py audit --out outputs/ocr/label-audit-v1
.venv-kie/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/label-audit-v1 --engine vietocr
.venv-ocr/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/label-audit-v1 --engine paddle
.venv/bin/python scripts/prepare_ocr_training.py finalize \
  --work outputs/ocr/label-audit-v1 --corrections configs/ocr_label_corrections_v1.json \
  --out data/processed/ocr_review_v1
.venv/bin/python scripts/prepare_ocr_training.py verify --root data/processed/ocr_review_v1
DVC_SITE_CACHE_DIR=/tmp/vietreceipt-dvc-state .venv/bin/dvc add data/processed/ocr_review_v1
```

Review artifacts local: `outputs/ocr/label-audit-v1/audit.json`, `visual_review.json`,
`review_candidates.json`, các ảnh `review_*.png`/`zoom_*.png` và prediction A/B.
DVC đã tạo cache local; chưa push remote.

## Fine-tune

VietOCR dùng checkpoint `vgg_transformer` đã cache, giữ vocabulary và preprocessing của adapter.
Fine-tune toàn model bằng AdamW, LR 1e-5, batch 4 × accumulation 4, clip grad 1,
label smoothing 0,1, tối đa 5 epoch, early stopping sau 2 epoch không cải thiện.
Resize bằng Pillow LANCZOS hiện tại, không dùng `Image.ANTIALIAS` đã bị bỏ.
Batch train gom theo chiều rộng, có đủ cả batch cuối; validation không padding khác chiều rộng.
Best checkpoint chọn bằng **corpus CER trên đủ 605 crop validation**, giữ dấu/hoa-thường sau NFC.

```bash
.venv-kie/bin/python scripts/train_vietocr.py --out outputs/training/vietocr-v1
```

Paddle dùng mã nguồn tag **v3.7.0**, commit `b03f46425e8ff4442b268ce449e3eef758146cd4`
và checkpoint huấn luyện chính thức `latin_PP-OCRv5_mobile_rec_pretrained.pdparams`.
Giữ kiến trúc MultiHead và loss CTC + NRTR của upstream; thay metric chọn checkpoint bằng corpus NFC CER.
Fine-tune toàn model, Adam/cosine LR 5e-5, batch 16, 5 epoch, eval mỗi epoch.
Train resize 48×640; không bật augmentation trong vòng đầu để kiểm tra khả năng thích nghi.
Nâng giới hạn text từ 25 lên 100 ký tự và kiểm tra từ điển để tránh bỏ dòng dài/ký tự âm thầm.
Validation dùng toàn bộ mẫu, batch 1, resize của upstream.

```bash
git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git models/training/PaddleOCR
.venv-ocr/bin/python -m pip install -r requirements-ocr-training.txt
.venv-ocr/bin/python scripts/train_paddle.py --out outputs/training/paddle-v1
.venv-ocr/bin/python models/training/PaddleOCR/tools/export_model.py \
  -c outputs/training/paddle-v1/training.yml \
  -o Global.pretrained_model=outputs/training/paddle-v1/best_accuracy \
     Global.save_inference_dir=outputs/training/paddle-v1/inference
```

Trainer Paddle tự tải training weights nếu chưa có. Hai trainer lưu data/model/config hashes và versions.
Trainer VietOCR còn có `--epochs 1 --max-batches 2` cho smoke test; đây không phải run đầy đủ.
Chạy lại bằng thư mục output mới để giữ lịch sử. Chưa khóa mọi dependency của hai môi trường.

## Đánh giá và chọn model

Chạy lại cả pretrained và fine-tuned trên cùng 605 PNG/nhãn đã review, cùng batch order `width`, batch 16.
Bundle evaluation dùng `config_file`/`weights` cho VietOCR và `model_dir` cho Paddle export.
`scripts/evaluate_ocr_training.py` kiểm tra cùng sample/ref/crop/group, tính corpus CER/WER,
bootstrap CI theo merchant/template và paired CI candidate−baseline:

```bash
.venv/bin/python scripts/evaluate_ocr_training.py \
  --works outputs/ocr/reviewed-val-pretrained outputs/ocr/reviewed-val-finetuned \
  --out outputs/ocr/finetune-summary
```

Tập 605 crop có sửa nhãn/hướng và loại crop thiếu transcription, nên **không so trực tiếp** số liệu này
với baseline E1/E2 trên 601 vùng GT. Việc ưu tiên review theo lỗi cũng ảnh hưởng tập validation mới.
Vì vậy cần báo thêm A/B E1/E2 trên **tham chiếu benchmark gốc không đổi**, cùng box/crop `val-none`;
chỉ recognizer thay đổi. Kiểm tra tiền/ngày/giờ sau chuẩn hóa và thời gian inference trước khi chọn model.
Tốc độ batch/crop không thay thế P95 API. Model chưa vượt qua benchmark cuối; test vẫn giữ đóng.

Nguồn kỹ thuật: [Paddle training config](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml),
[VietOCR](https://github.com/pbcquoc/vietocr).

## Kết quả vòng 1 trên benchmark gốc (2026-10-07)

`vietocr-v1` (LR 1e-5, 5 epoch): CER trên 605 crop đã review giảm **12,59% → 10,24%**, vẫn giảm ở epoch cuối.
Trên **tham chiếu benchmark gốc** (`val-none`, cùng box/crop, chỉ đổi weights):

| Run | E1 CER | E1 WER | E2 CER | E2 WER | TOTAL E2 | DATE E2 | TIME E2 |
|---|---:|---:|---:|---:|---|---|---|
| pretrained | 12,42% | 32,60% | 20,32% | 36,74% | 81/98 | 67/89 | 25/41 |
| vietocr-v1 | 10,70% | 29,91% | 19,52% | 35,09% | 77/98 | 65/89 | 27/41 |

ΔCER paired: E1 **−1,72** [−3,42; +0,06], E2 **−0,80** [−2,18; +0,69] điểm %. Lợi ích E1 chưa chuyển sang crop
do detector cắt, và accuracy giá trị TOTAL/DATE giảm. Chưa chọn v1 làm model.

## Vòng 2: chống học thuộc template

Train có 4.764 crop nhưng **79% thuộc 2 group** và chỉ 1.920 text khác nhau; 40 group nhỏ tổng cộng 260 crop.
Vì vậy không tách tập dev từ train (sẽ lấy mất phần lớn template hiếm); checkpoint vẫn chọn trên validation,
số liệu validation có bias chọn mẫu và test là trọng tài cuối.
Section `vietocr_v2` của `configs/ocr_finetune.yaml` bật thêm:

- **Sampling**: mỗi (group, text) tối đa 2 bản, P(group) ∝ n^0,7, rút 4.764 mẫu có hoàn lại mỗi epoch
  (group nhỏ từ ~5% lên ~32% mẫu).
- **Augmentation** OpenCV: pad, xoay ±2°, hạ phân giải, blur, contrast, noise, JPEG; không lật.
- **Decoder masking** 5% (như `masked_language_model` của VietOCR), **bf16 AMP**, batch 16 × 2,
  LR 1e-4 warmup 5% + cosine, tối đa 15 epoch, patience 4. `run.json` ghi commit/dirty của repo.

```bash
for lr in 1e-4 5e-5 2e-4; do
  .venv-kie/bin/python scripts/train_vietocr.py --section vietocr_v2 --learning-rate $lr \
    --out outputs/training/vietocr-v2-lr$lr
done
```

## Đánh giá checkpoint trên benchmark gốc

`fork` tạo run mới dùng **cùng crop/box/GT** của `val-none` (hard link, kiểm tra hash) với weights khác;
không ghi đè prediction baseline. VietOCR bắt buộc `--weights` để hash chính checkpoint; Paddle dùng `--model-dir`
(thư mục export). `compare_ocr_recognizers.py` tính E1/E2 CER/WER, paired CI theo merchant và accuracy TOTAL/DATE/TIME.

```bash
.venv/bin/python scripts/ocr_baseline.py fork --work outputs/ocr/val-none \
  --out outputs/ocr/val-none-vietocr-v1 --engine vietocr --weights outputs/training/vietocr-v1/best.pth
.venv-kie/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/val-none-vietocr-v1 --engine vietocr
.venv/bin/python scripts/ocr_baseline.py evaluate --work outputs/ocr/val-none-vietocr-v1 --engines vietocr
.venv/bin/python scripts/compare_ocr_recognizers.py --base outputs/ocr/val-none \
  --candidates outputs/ocr/val-none-vietocr-v1 --engine vietocr --out outputs/ocr/finetune-benchmark-v1
```

Chọn model theo **E2 và accuracy giá trị**, không chỉ CER crop đã review.

## Kết quả vòng 2 (2026-10-07)

Best checkpoint trên 605 crop review: LR 5e-5 **8,42%** (epoch 6), 1e-4 **8,63%** (epoch 6), 2e-4 **8,96%** (epoch 3);
CER dao động ±0,5–1 điểm % giữa các epoch. Trên benchmark gốc (`outputs/ocr/finetune-benchmark-v2`):

| Run | E1 CER | E1 WER | E2 CER | E2 WER | TOTAL E2 | DATE E2 | TIME E2 |
|---|---:|---:|---:|---:|---|---|---|
| pretrained | 12,42% | 32,60% | 20,32% | 36,74% | 81/98 | 67/89 | 25/41 |
| v1 (LR 1e-5) | 10,70% | 29,91% | 19,52% | 35,09% | 77/98 | 65/89 | 27/41 |
| v2 LR 5e-5 | 9,40% | 27,17% | 18,22% | 32,15% | 74/98 | 68/89 | 30/41 |
| **v2 LR 1e-4** | **9,23%** | **26,37%** | **18,01%** | **30,51%** | 79/98 | **70/89** | **30/41** |
| v2 LR 2e-4 | 9,63% | 27,67% | 18,23% | 32,75% | 81/98 | 69/89 | 30/41 |

v2 LR 1e-4 so với pretrained, ΔCER paired: E1 **−3,19** [−5,70; −0,60], E2 **−2,31** [−4,22; −0,49] điểm %;
ΔWER E2 **−6,23** [−10,59; −1,44]. Đây là vòng đầu tiên CI E2 không chứa 0. Vẫn xa mục tiêu CER 5%.

**TOTAL chưa đáng tin:** `parse_vnd` đọc sai số có dấu cách phân nhóm (`'64 000'` → 0, `'39 600'` → 600).
Hai trong ba vùng TOTAL "mất" của v2 LR 1e-4 là do lỗi parser hoặc định dạng (`64,000` đúng giá trị nhưng bị tính sai).
Cần sửa parser rồi tính lại bảng trước khi dùng TOTAL để chọn model.
