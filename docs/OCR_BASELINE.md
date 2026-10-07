# OCR baseline (M2)

Hai cấu hình dùng **cùng detection và cùng file crop PNG**:

- A: PaddleOCR detection + PaddleOCR recognition, chạy bằng `.venv-ocr`.
- B: PaddleOCR detection + VietOCR recognition, chạy bằng `.venv-kie`.

Mỗi lệnh chạy trong một process riêng, không import Paddle và Torch cùng process.
Model được chọn rõ trong `configs/ocr.yaml`, không dùng model mặc định của thư viện.
Weights/cache lưu trong `models/`, crop và báo cáo trong `outputs/`; cả hai thư mục đều bị Git ignore.

## Hai cách đánh giá

**E1 (recognition):** cắt crop bằng polygon GT từ manifest, rồi nhận dạng. Đo khả năng đọc chữ khi đã biết vị trí.
**E2 (detection + recognition):** detector tìm dòng trên ảnh đầy đủ, crop được lưu một lần cho cả A/B.
Box được map về tọa độ ảnh của manifest bằng homography nghịch đảo; ghép vào vùng GT theo độ phủ diện tích dòng dự đoán.
Vùng GT không có dòng khớp vẫn được tính với hypothesis rỗng (lỗi deletion).

MC-OCR chỉ gán nhãn bốn loại SELLER, ADDRESS, TIMESTAMP, TOTAL_COST, gồm cả vùng key có transcription.
E1/E2 báo cáo trên **các vùng được gán nhãn**, không phải transcript toàn trang. Dòng dự đoán ngoài vùng GT không thể kết luận là false positive.
`region_coverage` là tỷ lệ vùng GT có ít nhất một dòng được ghép, không phải detection recall/precision toàn trang.
Vùng quá nhỏ (cạnh dưới 8 px) hoặc text rỗng được loại và đếm trong `skipped_regions`.
E1 dùng polygon của manifest; bộ crop trong `recognition_*.tsv` hiện chưa phải đầu vào của runner này.

CER/WER được tính bằng tổng edit distance / tổng độ dài tham chiếu, sau Unicode NFC và gom whitespace.
`exact` giữ hoa/thường và dấu; `nocase`, `nodiac` chỉ hỗ trợ phân tích lỗi.
Báo cáo có CI 95% bằng bootstrap 1.000 lần theo `group_id` merchant/template và paired CI của A−B.
`p_a_better` là tỷ lệ mẫu bootstrap có A−B < 0, không phải p-value của kiểm định giả thuyết.

## Môi trường

Python 3.11. Cài PaddlePaddle GPU hoặc CPU phù hợp máy **trước** khi chạy; `requirements-ocr.txt` không tự chọn wheel Paddle.
Cài `.venv-kie` với Torch phù hợp CUDA và `requirements-kie.txt`.
Hai phiên bản adapter đã đối chiếu: PaddleOCR 3.7.0 và VietOCR 0.3.12.
Runner lưu phiên bản các thư viện thực tế vào mỗi run; các dependency còn lại chưa được khóa đầy đủ.

Nếu chạy CPU, thêm `--device cpu` cho cả prepare và recognize.
Thiết bị GPU dùng tên `gpu:0` cho Paddle, `cuda:0` cho VietOCR.
Lần chạy đầu cần mạng để tải model; VietOCR tải config từ repository tác giả và checkpoint từ URL trong config.
Có thể chạy offline bằng `detection.model_dir`, `recognition.paddle.model_dir`,
`recognition.vietocr.config_file` (YAML đầy đủ, đã gộp base) và `recognition.vietocr.weights` (checkpoint local).
VietOCR tắt việc tải pretrained backbone riêng vì checkpoint OCR đã chứa backbone.
Adapter dùng `Image.Resampling.LANCZOS` thay cho `Image.ANTIALIAS` đã bị Pillow mới loại bỏ,
giữ cách resize/chuẩn hóa và gom batch theo chiều rộng của VietOCR 0.3.12.
Runner mặc định sắp crop theo tỷ lệ rộng/cao trước khi chia batch (`--batch-order width`),
rồi khôi phục thứ tự sample trong prediction. Cách này giảm số batch rất nhỏ trong VietOCR;
không đổi pixel crop. `--batch-order input` giữ thứ tự gốc; lựa chọn được lưu trong metadata run.
Cách chia batch có thể thay đổi padding và prediction của recognizer; benchmark chính thức cố định `width` cho cả A/B.
Runner dừng nếu yêu cầu GPU Paddle nhưng GPU không khả dụng, tránh ghi metadata GPU cho lần chạy tự chuyển sang CPU.

## Smoke test rồi benchmark

Chạy từ gốc repository. Bắt đầu với 10 ảnh validation và variant `none`:

```bash
.venv-ocr/bin/python scripts/ocr_baseline.py prepare --out outputs/ocr/val-none-smoke --limit 10
.venv-ocr/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/val-none-smoke --engine paddle
.venv-kie/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/val-none-smoke --engine vietocr
.venv/bin/python scripts/ocr_baseline.py evaluate --work outputs/ocr/val-none-smoke
```

Chạy toàn bộ validation bằng thư mục mới, bỏ `--limit`. Lặp lại với `--preproc rules` và `--preproc full` để đo ablation.
Có thể chỉ tạo GT crop mà không cần detector/model/network bằng `.venv/bin/python ... prepare --mode e1`.
`--mode e2` chỉ tạo crop từ detection. Thư mục prepare phải mới/rỗng để tránh trộn run.
Nhận dạng và đánh giá từng engine độc lập được hỗ trợ:

```bash
.venv/bin/python scripts/ocr_baseline.py evaluate --work outputs/ocr/val-none-smoke --engines paddle
```

Mặc định chỉ dùng validation; test/test_seen yêu cầu cả `--split test` (hoặc `test_seen`) và `--allow-test` khi tạo run.
Chỉ mở test sau khi chốt model, preprocessing và ngưỡng ghép vùng trên validation.
Thêm `--log-mlflow` ở bước evaluate để gửi báo cáo vào experiment `ocr-baseline`, lấy tracking URI từ Settings.

Sau khi cả A/B đã được evaluate ở mỗi variant, tổng hợp bảng so sánh, paired CI giữa các variant
và giá trị tiền/ngày/giờ sau chuẩn hóa (thêm `--plot` nếu môi trường có matplotlib):

```bash
.venv/bin/python scripts/summarize_ocr_validation.py \
  --works outputs/ocr/val-none outputs/ocr/val-rules outputs/ocr/val-full \
  --out outputs/ocr/validation-summary --plot
```

Script kiểm tra các run đều thuộc validation và có cùng manifest/tập tham chiếu.

## Ablation hướng đọc 180°

Lệnh `orientation` dùng VietOCR trên tối đa 12 crop detection E2 lớn nhất đủ điều kiện mỗi ảnh,
đọc cùng batch ở 0° và 180°. Không đọc text/box GT, nhãn, role hoặc quality để chọn hướng.
Chỉ xoay khi có ít nhất 5 cặp confidence hợp lệ, median chênh lệch ≥ 0,05,
ít nhất 70% dòng nghiêng về 180° và median confidence sau xoay ≥ 0,8.
Đây là heuristic chưa calibration; trường hợp thiếu bằng chứng giữ nguyên ảnh.
Ngưỡng trong `configs/ocr.yaml:orientation` phải cố định trước khi xem kết quả ablation.

```bash
.venv-kie/bin/python scripts/ocr_baseline.py orientation \
  --work outputs/ocr/val-none --out outputs/ocr/orientation-val-v1.json
.venv-ocr/bin/python scripts/ocr_baseline.py prepare \
  --out outputs/ocr/val-orient180 --preproc orient180 \
  --orientation outputs/ocr/orientation-val-v1.json
.venv-ocr/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/val-orient180 --engine paddle
.venv-kie/bin/python scripts/ocr_baseline.py recognize --work outputs/ocr/val-orient180 --engine vietocr
.venv/bin/python scripts/ocr_baseline.py evaluate --work outputs/ocr/val-orient180
.venv/bin/python scripts/summarize_ocr_validation.py \
  --works outputs/ocr/val-none outputs/ocr/val-orient180 \
  --out outputs/ocr/orientation-summary --plot
```

`orientation` chỉ nhận run `none` có E2 thuộc train/val và yêu cầu file output mới.
File quyết định lưu các cặp dự đoán/confidence, ngưỡng, model config/device/versions,
hash dataset nguồn, manifest, crop và ảnh gốc. `prepare` kiểm tra manifest/split,
đủ quyết định cho tập ảnh cần chạy và hash ảnh trước khi nạp detector.
Variant `orient180` bắt buộc file quyết định; các variant khác không nhận file này.
Ảnh đã xoay được detect và crop lại; A/B vẫn dùng chung PNG.
Homography ánh xạ box về ảnh gốc để ghép GT; ghép text theo thứ tự đọc trong ảnh đã sửa hướng.

Thời gian probe và detection ban đầu là chi phí bổ sung, được lưu riêng trong file quyết định,
**không nằm trong `e2_compute_s`**. Ablation này chưa đo latency toàn pipeline hoặc chất lượng
classifier hướng trên tập có nhãn hướng. Test/test_seen vẫn chưa được mở cho vòng cải thiện này.

## Đầu ra

| File | Nội dung |
|---|---|
| `dataset.json` | Snapshot config, hash manifest, vùng GT, box detection, danh sách crop và hash từng PNG |
| `crops/*.png` | Crop dùng chung; hash được kiểm tra trước mỗi lần nhận dạng |
| `predictions_<engine>.jsonl` | Text, confidence và thời gian amortized theo crop |
| `pages_<engine>.jsonl` | Toàn bộ dòng E2 với text, confidence và polygon ở tọa độ ảnh manifest |
| `run_<engine>.json` | Model config, device, phiên bản thư viện, hash dataset/predictions và thời gian batch |
| `report.json` | CER/WER, CI, exact match, coverage, kết quả theo trường, paired differences |
| `errors_<engine>.csv` | Toàn bộ cặp ref/hyp và số lỗi để lọc, phân tích |

`e2_compute_s` là ước lượng preprocessing + detection + recognition amortized theo batch, bao gồm mọi dòng được detect.
Chỉ số này loại thời gian khởi tạo model và I/O, chưa warmup riêng; không dùng để khẳng định P95 API <= 5 giây.
Không so sánh độ tự tin của hai recognizer như xác suất đã calibration.

## Tiêu chí hoàn thành baseline

Benchmark local (2026-10-07) đã hoàn tất A/B × `none`/`rules`/`full` trên toàn bộ 113 ảnh validation,
601 vùng hợp lệ. Crop/prediction/reference được audit và edit distance được đối chiếu bằng thuật toán độc lập.
Kết quả, giới hạn và bước cải thiện: [OCR_VALIDATION_RESULTS.md](OCR_VALIDATION_RESULTS.md).
VietOCR + `none` đạt E1 CER 12,42%, E2 CER 20,32%; chưa đạt mục tiêu chất lượng của kế hoạch.

Smoke test A/B chạy thành công trên cùng crop; có báo cáo validation cho các variant, ví dụ lỗi và log MLflow.
Sau đó chốt cấu hình, đánh giá test/test_seen riêng. CER <= 5% và WER <= 12% là mục tiêu của kế hoạch,
không phải chất lượng được đảm bảo bởi model pretrained. Chỉ quyết định fine-tune khi có số liệu và phân tích lỗi.

Tài liệu adapter: [Paddle detection](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html),
[Paddle recognition](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html),
[VietOCR](https://github.com/pbcquoc/vietocr).
