# Ablation hướng đọc 0°/180° trên validation

Ngày chạy: 2026-10-07. Đã thêm probe hướng đọc, tạo lại detection/crop dùng chung và benchmark
Paddle/VietOCR trên đủ **113 ảnh, 39 nhóm merchant/template, 601 vùng GT hợp lệ**.
VietOCR giảm E1 CER từ **12,42% → 12,24%**, E2 CER từ **20,32% → 20,14%**.
Mức giảm nhỏ, CI paired còn chạm 0 và heuristic bỏ sót một ảnh lộn đã xác nhận.
**Giữ `none` làm mặc định; `orient180` là variant thử nghiệm.**

## Phương pháp

- Nguồn probe: crop detection E2 của `val-none`. VietOCR đọc cả 0° và 180°,
  tối đa 12 crop lớn nhất mỗi trang có cạnh ≥ 12 px và tỷ lệ rộng/cao trong [2; 25].
- Quyết định chỉ dựa vào pixel crop và confidence. Không dùng transcription/box GT,
  nhãn trường, role, quality hoặc metric validation để chọn hướng.
- Ngưỡng cố định trước inference: ≥ 5 cặp có text/confidence hợp lệ,
  median(score180 − score0) ≥ 0,05, ≥ 70% phiếu chọn 180°, median score180 ≥ 0,8.
  Đã giữ nguyên ngưỡng sau khi xem kết quả; đây chưa phải xác suất calibration.
- Có **1.356 cặp probe**; chọn lật **1/113 ảnh**. Run mới xoay ảnh trước detection,
  crop lại một lần rồi dùng chung PNG cho A/B. Homography 180° dùng tâm pixel `(w−1, h−1)`;
  box được map về frame ảnh manifest để ghép GT, còn thứ tự ghép text dùng frame sau xoay.
- Cùng model/batch 16/order `width`, normalization NFC, ngưỡng ghép vùng và tham chiếu với baseline.
  CI 95% bootstrap theo merchant/template, 1.000 lần, seed 42. Test/test_seen chưa chạy.

## Kết quả

| Variant | Recognizer | E1 CER | E1 WER | E2 CER | E2 WER | E2 coverage |
|---|---|---:|---:|---:|---:|---:|
| none | Paddle | 15,15% | 48,55% | 23,60% | 53,04% | 94,34% |
| orient180 | Paddle | 15,04% | 48,50% | 23,49% | 53,04% | 94,34% |
| none | VietOCR | 12,42% | 32,60% | 20,32% | 36,74% | 94,34% |
| orient180 | VietOCR | **12,24%** | **32,40%** | **20,14%** | **36,59%** | 94,34% |

VietOCR `orient180` có CI CER E1 **[7,41%; 17,45%]**, E2 **[11,07%; 28,82%]**.
Chênh lệch paired `orient180 − none`, điểm phần trăm:

| Recognizer | E1 ΔCER [CI 95%] | E2 ΔCER [CI 95%] |
|---|---:|---:|
| Paddle | −0,10 [−0,42; +0,06] | −0,10 [−0,45; +0,05] |
| VietOCR | −0,18 [−0,72; 0,00] | −0,18 [−0,72; 0,00] |

Coverage giữ **567/601 vùng**. Accuracy E2 TOTAL/DATE/TIME của VietOCR giữ nguyên
**81/98**, **67/89**, **25/41**. Bởi vậy chưa có bằng chứng cải thiện trích xuất giá trị nghiệp vụ.
Paddle TIME giảm 29/41 → 28/41 trong lần chạy này.

![So sánh CER và CI](../outputs/ocr/orientation-summary/cer_comparison.png)

## Kiểm tra trực quan và giới hạn

**`mcocr_public_145014eeksl.jpg`** được chọn lật: median margin **0,1224**, 10/12 phiếu 180°,
median confidence tăng **0,7700 → 0,9059**. Ảnh biên lai thực tế nằm ngang 90°;
lật 180° sửa hướng đọc của crop dọc theo quy ước xoay 90° hiện có, chưa đưa cả trang về chiều đứng.
Ảnh này chỉ có **một vùng GT**, là key `TỔNG TIỀN PHẢI T.TOÁN`.
VietOCR giảm từ **18 xuống 1** phép sửa ở cả E1/E2; E1 đọc `TỔNG TIẾN PHẢI T.TOÁN`,
E2 đọc `TỔNG TIỀN PHẢI T. TOÁN`. Cải thiện chủ yếu nằm ở một key, chưa giải quyết các field value.

**`mcocr_public_145014smasw.jpg`** vẫn bị bỏ sót dù ảnh biên lai lộn 180° đã được xác nhận.
Watermark Samsung đúng chiều chiếm ba crop lớn trong bộ probe và có confidence cao khi giữ nguyên;
nhiều dòng sai hướng khác cũng được VietOCR đọc với confidence cao. Median margin chỉ **0,0008**,
phiếu 180° **6/12**. Điều này cho thấy confidence recognizer đơn lẻ chưa đủ làm classifier hướng.
Chưa benchmark 4 hướng 0°/90°/180°/270° hoặc có nhãn hướng để đo accuracy classifier.

112 ảnh không lật có **6.373 crop giữ nguyên hash/pixel**. Tuy vậy **46 prediction Paddle và 1 prediction
VietOCR** đổi text trên chính những crop này. Lần detect mới có 63 dòng ở trang được lật thay vì 61,
làm thay đổi nhóm batch theo chiều rộng. Vì vậy chênh lệch toàn corpus bao gồm cả ảnh hưởng batching/inference;
không quy mọi thay đổi prediction cho rotation.

Probe tốn **154,19 giây inference** tổng cộng, chưa gồm khởi tạo/I/O.
Detection nguồn và hai lượt recognition probe là chi phí thêm, không nằm trong `e2_compute_s`.
Chưa đo latency toàn pipeline và chưa kết luận đạt P95 ≤ 5 giây.

## Kiểm chứng và artifacts

- **90 test pass**, Ruff và `git diff --check` sạch. Test mới kiểm tra confidence thiếu/mơ hồ,
  input bị sửa, guard train/val, hash ảnh khi replay, pixel/homography 180° và thứ tự đọc nhiều dòng sau xoay.
- Audit cả baseline và candidate: **12.872 hash crop**, **4.808 cặp ref/hyp** được kiểm tra;
  edit distance CER/WER tính lại bằng dynamic programming độc lập khớp báo cáo.
  Toàn bộ vùng GT khớp manifest gốc; hash dataset/predictions khớp metadata.
- Với 63 box E2 của trang được lật, đã tái dựng crop từ ảnh gốc và box map ngược.
  Kích thước đều khớp, sai số pixel tối đa 3/255, MAE lớn nhất 0,244/255 do rounding OpenCV
  khi tính lại `minAreaRect` từ tọa độ float. Round-trip homography khớp.
- Manifest SHA256: `ddd630c9baa59798ecde31c10ff711652f7c6c7176b72e9aceb775f2af6a7cfd`.
- GPU RTX 3060 Laptop 6 GB, driver 617.14. Code/config/model hash:
  [orientation_environment.json](../outputs/ocr/orientation_environment.json).
- [Quyết định và bằng chứng probe](../outputs/ocr/orientation-val-v1.json),
  [run A/B](../outputs/ocr/val-orient180/report.json),
  [so sánh/paired CI](../outputs/ocr/orientation-summary/comparison.json),
  [audit](../outputs/ocr/orientation-summary/integrity_audit.json).
  Artifacts nằm local trong `outputs/`, được Git ignore; chưa gửi MLflow từ xa.

Lệnh inference và tái tạo so sánh: [OCR_BASELINE.md](OCR_BASELINE.md).
Audit local có thể chạy lại từ gốc repo:

```bash
PYTHONPATH=. .venv/bin/python outputs/ocr/orientation-summary/audit_ocr_orientation.py
```

## Bước tiếp theo

Giữ VietOCR + `none` và báo cáo baseline làm mốc. Cần kiểm tra nhãn/role trước fine-tune
(các đề xuất trong `annotation_review.json` vẫn chưa được áp dụng vào GT).
Vòng xử lý hướng sau nên đánh giá classifier 4 hướng hoặc classifier hướng text-line,
có biện pháp hạn chế watermark và tập kiểm tra hướng được review; heuristic hiện tại chưa đủ để bật mặc định.
Sau audit nhãn, chuẩn bị train text-line crops và fine-tune VietOCR, chọn checkpoint bằng validation E1,
theo dõi E2. Chưa chốt model/preprocessing nên tiếp tục giữ test đóng.
