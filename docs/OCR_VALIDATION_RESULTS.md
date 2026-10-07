# Kết quả OCR baseline trên toàn bộ validation

Ngày chạy: 2026-10-07. Cấu hình có kết quả tổng thể tốt nhất trong benchmark này là **Paddle detection + VietOCR, preprocessing `none`**. E1 CER đạt **12,42%**, E2 CER đạt **20,32%**. Cả hai recognizer chưa đạt mục tiêu CER ≤ 5% và WER ≤ 12%; chưa đủ cơ sở chuyển sang đánh giá test cuối cùng.

## Phạm vi và cách đo

- Toàn bộ **113 ảnh validation**, **39 nhóm merchant/template**, **601 vùng GT hợp lệ**. Loại **13/614 vùng** có cạnh dưới 8 px hoặc text rỗng theo cấu hình hiện tại.
- A: `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec`; B: cùng detector và cùng file PNG crop + VietOCR `vgg_transformer`.
- E1 nhận dạng crop polygon GT. E2 detect trên ảnh, map box về tọa độ gốc rồi ghép vào vùng GT với overlap tối thiểu 0,5. Vùng không được ghép vẫn tính lỗi deletion.
- MC-OCR chỉ có transcription các vùng của bốn trường. **E2 không phải CER toàn trang**; không thể dùng các dòng ngoài GT để tính false positive hay precision/recall toàn trang.
- CER/WER là tỷ lệ tổng edit distance trên tổng độ dài tham chiếu, giữ hoa/thường và dấu sau NFC + gom whitespace. CI 95% bootstrap theo nhóm merchant/template, 1.000 lần, seed 42.
- `none`: giữ nguyên ảnh; `rules`: deskew + CLAHE + denoise; `full`: thêm perspective correction. Giữ cùng tập tham chiếu cho cả ba variant.
- GPU RTX 3060 Laptop 6 GB, batch size 16, batch order `width` cho cả hai recognizer. Thay đổi batch có thể thay đổi padding và output; số liệu smoke/input-order trước đó không được trộn vào bảng này.
- Test/test_seen chưa được chạy OCR. Nhãn và tham chiếu benchmark được giữ nguyên.

## Kết quả

| Preprocessing | Recognizer | E1 CER | E1 WER | E2 CER | E2 WER | E2 region coverage |
|---|---|---:|---:|---:|---:|---:|
| none | Paddle | 15,15% | 48,55% | 23,60% | 53,04% | 94,34% |
| none | VietOCR | **12,42%** | **32,60%** | **20,32%** | **36,74%** | 94,34% |
| rules | Paddle | 15,12% | 48,06% | 23,95% | 53,59% | 94,18% |
| rules | VietOCR | 12,95% | 34,10% | 21,65% | 40,08% | 94,18% |
| full | Paddle | 15,06% | 47,66% | 23,85% | 53,29% | 94,18% |
| full | VietOCR | 13,25% | 34,75% | 22,40% | 40,98% | 94,18% |

`none` có 567/601 vùng được ghép; `rules` và `full` có 566/601. Coverage chỉ mô tả ghép vùng có nhãn, không đánh giá đầy đủ khả năng tìm chữ trên ảnh.

![CER và CI 95%](../outputs/ocr/validation-summary/cer_comparison.png)

Với `none`, VietOCR E1 CER có CI **[7,60%; 17,62%]**, E2 CER có CI **[11,33%; 28,99%]**. So sánh paired Paddle−VietOCR:

| Metric | Chênh lệch, điểm phần trăm | CI 95% |
|---|---:|---|
| E1 CER | +2,73 | [-0,16; +5,85] |
| E2 CER | +3,28 | [+1,52; +5,23] |
| E1 WER | +15,95 | [+8,97; +21,61] |
| E2 WER | +16,30 | [+9,77; +21,21] |

VietOCR có CER thấp hơn theo giá trị đo và có bằng chứng paired ở E2; CI E1 CER còn chứa 0 nên chưa kết luận chắc chắn về lợi thế E1 CER. VietOCR cũng không thắng mọi metric nghiệp vụ: nhận dạng giờ sau chuẩn hóa ở E2 kém Paddle.

Tiền xử lý chưa cho thấy cải thiện tổng thể. Với VietOCR, `rules` tăng E2 CER 1,33 điểm % so với `none`, CI [-0,81; +3,39]; `full` tăng 2,08 điểm %, CI [-0,04; +4,24]. Cả hai CI E2 còn chứa 0. Riêng E1 của `full` tăng 0,83 điểm %, CI [+0,31; +1,61]. Perspective được áp dụng ở 54/113 ảnh; chưa có lý do bật variant này mặc định.

## Chất lượng theo trường và ảnh

VietOCR + `none`:

| Trường | Số vùng | E1 CER | E2 CER | E2 coverage |
|---|---:|---:|---:|---:|
| SELLER | 111 | 21,29% | 23,09% | 96,40% |
| ADDRESS | 113 | 10,15% | 22,02% | 93,81% |
| TOTAL_COST | 193 | 11,83% | 14,65% | 97,93% |
| TIMESTAMP | 184 | 10,08% | 21,81% | 89,67% |

Nhóm chất lượng thấp (`quality < 0,5` theo manifest) có 25 ảnh/113 vùng: E1 CER **27,13%**, E2 CER **33,83%**. Nhóm còn lại có 88 ảnh/488 vùng: **9,11%** và **17,28%**. Đây là tín hiệu cần ưu tiên kiểm tra ảnh khó và hướng ảnh trước khi quy toàn bộ sai số cho model.

E1 CER của VietOCR giảm từ 12,42% xuống 9,98% khi bỏ khác biệt hoa/thường, và 8,28% khi tiếp tục bỏ dấu. Những biến thể này chỉ dùng phân tích; không thay thế metric exact hoặc mục tiêu chất lượng.

Giá trị E2 sau chuẩn hóa, chỉ trên các vùng `value` có tham chiếu parse được:

| Recognizer, `none` | TOTAL VND | DATE | TIME |
|---|---|---|---|
| Paddle | 78/98 (79,59%) | 67/89 (75,28%) | 29/41 (70,73%) |
| VietOCR | 81/98 (82,65%) | 67/89 (75,28%) | 25/41 (60,98%) |

Đây là accuracy từng vùng, chưa phải accuracy trích xuất đúng toàn bộ chứng từ. Role key/value đang được suy ra bằng heuristic và có nhiễu annotation; không nên dùng bảng này để khẳng định chất lượng KIE.

## Kiểm tra triển khai và lỗi đáng chú ý

- **71 test pass**, Ruff và `git diff --check` sạch. Đã sửa tương thích Pillow/VietOCR, không cho Paddle âm thầm fallback CPU khi yêu cầu GPU, xử lý thứ tự góc quad ở 45° và giữ đúng sample ID sau sắp batch theo chiều rộng.
- Audit toàn benchmark xác nhận đủ 113 ảnh validation trong mỗi variant, cùng tham chiếu với manifest, toàn bộ **19.710 hash crop** hợp lệ, đủ một prediction cho mỗi crop ở cả A/B, hash dataset/prediction khớp metadata.
- Tính lại edit distance bằng dynamic programming độc lập với RapidFuzz cho **7.212 cặp ref/hyp** của sáu cấu hình E1/E2; CER/WER tổng hợp khớp báo cáo. Chi tiết: [integrity_audit.json](../outputs/ocr/validation-summary/integrity_audit.json).
- Đã kiểm tra trực quan `mcocr_public_145014smasw.jpg`: ảnh gốc và GT crop bị lộn 180°. `deskew` chỉ sửa nghiêng nhỏ nên không giải quyết trường hợp này. E1 SELLER có 24 phép sửa; cần benchmark riêng xử lý hướng đọc.
- `mcocr_public_145014wmrqi.jpg`, ADDRESS region 2: E2 hypothesis chứa thêm dòng mã số thuế, điện thoại và email; 63 phép sửa. Đây là vấn đề phạm vi vùng/ghép hình học, không thể coi toàn bộ là lỗi đọc chữ.
- Đã kiểm tra trực quan `mcocr_public_145013qrpce.jpg`: region 6 có GT `04 04`/TOTAL_COST nhưng crop hiện chữ `04`, nằm ở dòng số quầy; `TÊN ĐẠI LÝ` và `Địa chỉ:` cũng đang mang role `value`. Ghi đề xuất review trong [annotation_review.json](../outputs/ocr/annotation_review.json), **chưa sửa tham chiếu benchmark**.

Thời gian E2 trong `report.json` là ước lượng compute theo batch, không gồm khởi tạo/I/O và chưa đo warmup riêng. Không dùng số này để kết luận API đạt P95 ≤ 5 giây. Các dependency thực tế được ghi lại nhưng chưa có lockfile đầy đủ.

## Quyết định cho bước tiếp theo

Giữ **VietOCR + `none`** làm mốc so sánh cho vòng cải thiện tiếp theo, giữ Paddle làm đối chứng. Ưu tiên:

1. Rà soát các lỗi GT/role và nhóm lỗi E2 ghép dư hoặc bỏ vùng; nếu sửa nhãn, version manifest mới và giữ báo cáo cũ để so sánh.
2. Thêm xử lý hướng 180° và benchmark ablation trên validation, map box bằng homography như các bước hình học hiện có; không chọn hướng bằng nội dung GT.
3. Sau audit, fine-tune VietOCR trên **train text-line crops**, chọn checkpoint bằng validation E1 và theo dõi E2. Cả hai baseline vượt ngưỡng CER 5% nên đáp ứng điều kiện xem xét fine-tune của kế hoạch.
4. Chỉ mở test/test_seen sau khi chốt model, preprocessing và cách ghép vùng. Benchmark này chưa chạy fine-tune hoặc đánh giá test.

## Artifacts và tái tạo báo cáo

- [Tổng hợp đầy đủ và CI](../outputs/ocr/validation-summary/comparison.md), [JSON](../outputs/ocr/validation-summary/comparison.json).
- Run: [none](../outputs/ocr/val-none/report.json), [rules](../outputs/ocr/val-rules/report.json), [full](../outputs/ocr/val-full/report.json). Mỗi thư mục có crop, prediction, pages, errors CSV, metadata và log.
- [GPU, code/model SHA256](../outputs/ocr/benchmark_environment.json). Manifest SHA256: `ddd630c9baa59798ecde31c10ff711652f7c6c7176b72e9aceb775f2af6a7cfd`.
- Các artifacts trong `outputs/`/`models/` được Git ignore; tài liệu này lưu snapshot số liệu trong repository. Báo cáo giữ local, chưa gửi vào MLflow từ xa.

```bash
.venv/bin/python scripts/summarize_ocr_validation.py \
  --works outputs/ocr/val-none outputs/ocr/val-rules outputs/ocr/val-full \
  --out outputs/ocr/validation-summary --plot
```

Lệnh chạy inference mới và cách cài môi trường: [OCR_BASELINE.md](OCR_BASELINE.md).
