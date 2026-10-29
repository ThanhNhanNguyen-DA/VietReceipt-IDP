# Luồng đầu tiên trên ảnh MC-OCR

Một luồng chạy trọn từ ảnh MC-OCR đến JSON đã chuẩn hoá, có chấm điểm theo GT. Recognizer chính là **VietOCR** (`vgg_transformer`, pretrained); detector là PaddleOCR.

```
ảnh -> tiền xử lý (none) -> Paddle det -> VietOCR rec -> trích trường (luật) -> chuẩn hoá -> kiểm tra -> routing -> JSON + báo cáo
       .venv-ocr (prepare)                .venv-kie (recognize)        .venv (src/extraction, src/pipelines)
```

## Chạy

```bash
.venv/bin/python scripts/run_receipt_flow.py run --limit 10 --out outputs/flow/val-smoke  # 10 ảnh val
.venv/bin/python scripts/run_receipt_flow.py run --out outputs/flow/val                   # toàn bộ val (113 ảnh)
.venv/bin/python scripts/run_receipt_flow.py extract --work outputs/ocr/val-none --out /tmp/vietreceipt-val-extract
```

`run` gọi `scripts/ocr_baseline.py prepare/recognize` qua subprocess ở hai môi trường riêng (Paddle và Torch xung đột CUDA), rồi trích trường ở môi trường hiện tại.
`test` bị khoá: `--split` chỉ nhận `train`/`val`.

Đầu ra mặc định trong thư mục run: `extraction_vietocr.jsonl` (một dòng mỗi ảnh), `flow_rows_vietocr.jsonl` (điểm từng ảnh), `flow_report_vietocr.json`.
`extract --out <thư mục>` ghi cả ba file sang nơi khác, giữ nguyên run nguồn. `extract`, `evaluate` và hiệu chỉnh đều chặn run có split `test`/`test_seen`.
Mỗi dòng của `extraction_vietocr.jsonl` có `draft` (theo `ReceiptDraft`), `fields` (giá trị, độ tin cậy, luật, dòng OCR làm bằng chứng), `doc_confidence`, `routing`, `validation_issues`, `ocr.mean_rec_score`.

## Trích trường bằng luật (`src/extraction/rules.py`)

| Trường | Luật |
|---|---|
| `total` | dòng có từ khoá 'tổng cộng/thanh toán/phải trả'; số tiền cùng dòng, cùng hàng bên phải hoặc hàng liền dưới. Loại tiền khách đưa/trả lại/giảm giá/tổng kết. Không có từ khoá thì lấy số lớn nhất (độ tin cậy thấp) |
| `date`, `time` | regex ngày (`parse_date`) trong khoảng 2000..hôm nay; ưu tiên dòng có 'ngày'. Giờ ở cùng dòng/hàng với ngày, bỏ giờ mở cửa |
| `merchant_name` | chọn trong 8 dòng đầu bằng từ thương hiệu, chữ hoa và cạnh ngắn của box (không nhầm box xoay dọc với font lớn); gộp tối đa 3 mảnh tiêu đề gần nhau/cùng cỡ. Cắt hậu tố sau dấu gạch khi có cấu trúc địa chỉ; hậu tố chi nhánh chỉ bỏ nếu lặp trong địa chỉ gần đó |
| `merchant_address` | 1-3 dòng có từ khoá hoặc cấu trúc số nhà/phường trong 12 dòng đầu; hỗ trợ giá trị ngay sau nhãn `Địa chỉ:` tách riêng; loại liên hệ, ngày, mã hóa đơn và các mảnh tên đã gộp |

Số tiền chỉ nhận khi đứng riêng (không dính mã/ngày/số âm). Nhóm phân cách bằng dấu cách ('35 000', '1 234 567') được giữ riêng với nhóm dấu chấm/phẩy:
`Tổng số 5 116,000` lấy 116000, không ghép số lượng 5 vào tiền. Khoảng trắng quanh dấu phân cách trước đủ 3 chữ số được chuẩn hoá (`55, 000` → 55000).
Số liền không chia hết cho 100 được nhận khi đứng ngay sau nhãn tổng/thanh toán hoặc sát ký hiệu tiền; các dòng điện thoại bị loại.
Từ khoá loại người bán/tổng/địa chỉ dùng ranh giới từ hoặc cấu trúc cụ thể: `Hotel`, `Intel`, `Giò chả` không bị loại nhầm.
Ngày ưu tiên mẫu có năm đủ 4 chữ số rồi mới năm 2 chữ số. Chuỗi `18/03/10/2020` lấy phần hợp lệ `03/10/2020`; không thể khẳng định đó là ngày gốc dự định.
Độ tin cậy trường = độ tin cậy luật x `rec_score` của dòng chứa giá trị; `doc_confidence` = min theo `merchant.name`, `transaction.date`, `total`.
MC-OCR không có nhãn dòng hàng nên `items` không nằm trong trường bắt buộc của luồng này.
LayoutXLM (M3) sẽ thay `extract_fields` qua cùng đầu ra `dict[str, FieldResult]`, và baseline luật là mức nó phải vượt.

## Kết quả trên validation (113 ảnh, 39 nhóm; CI 95% bootstrap theo nhóm)

Đo lại ngày 2026-10-10 trên cùng OCR `val-none`, không chạy OCR và không sửa GT. `rules-v2` sửa A1–A8, B1 và một phần B4.

| Trường | Trước (`rules-v1`) | Sau (`rules-v2`) | CI 95% sau | CER trước → sau (không phân biệt hoa/thường) |
|---|---|---|---|---|
| Tổng tiền | 74/94 | 75/94 = 79,8% | [65,3; 91,8] | |
| Ngày | 70/81 | 71/81 = 87,7% | [78,3; 94,4] | |
| Giờ | 24/39 | 24/39 = 61,5% | [44,4; 80,9] | |
| Tên người bán | 62/94 | 68/94 = 72,3% | [55,7; 85,9] | 31,6% → 27,4% |
| Địa chỉ | 57/94 | 66/94 = 70,2% | [51,3; 81,7] | 33,9% → 27,3% |
| Ảnh đúng cả tổng, ngày, tên | 58/110 | 67/110 = 60,9% | [48,1; 74,7] | |

Đúng nghĩa là: tổng/ngày/giờ khớp một giá trị GT sau chuẩn hoá; tên/địa chỉ có CER <= 25% so với GT (tên: vùng gần nhất, địa chỉ: các vùng nối theo thứ tự đọc).
Số liệu OCR 84/98 trong `docs/OCR_VALIDATION_RESULTS.md` là snapshot trước vòng sửa này; không tái tính benchmark OCR để so với bảng trích trường.

**Cách đọc các con số:**
- Luật được chỉnh sau khi xem lỗi trên validation (tập phát triển), nên số này lạc quan hơn trên dữ liệu mới. Test vẫn khoá cho lần chấm cuối.
  Mẫu train 150 ảnh thử trước đó không dùng được để phát triển vì 79% thuộc 2 merchant (tổng tiền 90% ở đó, 46,8% trên val trước khi sửa).
- Phần lớn lỗi còn lại đến từ OCR/detector ('2820' thay vì '2020', '15' thay vì '16', dòng bị tách), không phải từ luật, đúng với khoảng cách E1→E2 đã đo.
- Nhãn GT của tên người bán không nhất quán ('Giảm giá', 'Tiền trả', 'TÊN ĐẠI LÝ:' được gán SELLER) nên CER tên bị kéo lên.

So từng ảnh, tên có 11 ca sai → đúng và 5 ca đúng → sai; địa chỉ có 9 ca sai → đúng và không có ca đúng → sai; ảnh đúng cả ba trường có 12 ca cải thiện, 3 ca giảm.
Năm ca tên giảm gồm: một ca `WINMART` → `Payoo` do logo nhà cung cấp thanh toán nổi bật hơn; hai ca GT chứa `HAU GIANG` nhưng luật B1 bỏ hậu tố chi nhánh;
hai ca GT là tiêu đề `TÊN ĐẠI LÝ` mà luật chọn logo. Không vá riêng các ảnh này: vẫn cần phân biệt logo dịch vụ/người bán và thống nhất nhãn tên chi nhánh.

## Routing và hiệu chỉnh doc_confidence

`doc_confidence` là xác suất ảnh đúng mọi trường bắt buộc: mỗi trường (tên, ngày, tổng) có một hồi quy logistic nhỏ trên tín hiệu thô
(độ tin cậy luật, `rec_score`, số ứng viên đồng ý, tổng có phải số lớn nhất, ...), rồi lấy **min** theo trường bắt buộc (kế hoạch mục 5).
Mô hình nằm ở `configs/doc_confidence_calibration.json`; thiếu file thì luồng dùng độ tin cậy thô.
Trường bắt buộc và mục tiêu lỗi lấy từ `configs/pipeline.yaml` (`required_fields`, `target_auto_accept_error`).

```bash
.venv/bin/python scripts/calibrate_confidence.py --runs outputs/ocr/val-none outputs/flow/calib-train
# Thử mà không ghi đè mô hình đang dùng:
.venv/bin/python scripts/calibrate_confidence.py --runs outputs/ocr/val-none outputs/flow/calib-train --out /tmp/vietreceipt-calibration.json
```

**Dữ liệu hiệu chỉnh:** 194 ảnh, 70 nhóm merchant = val + 85 ảnh train lấy tối đa 12 ảnh mỗi nhóm (không thuộc 150 ảnh train đã dùng để chỉnh luật).
Train có 832 ảnh nhưng 94% thuộc 4 merchant, nên không có tập hiệu chỉnh sạch lớn hơn; test vẫn khoá.
Script trích đặc trưng thô trong bộ nhớ, không ghi đè `extraction_vietocr.jsonl` hay các file khác trong run.
Khi chạy extract trên run đã khớp mô hình (metadata `runs`/`run_paths`), luồng in cảnh báo **in-sample**.
Mọi chỉ số chất lượng dưới đây là **out-of-fold**: cross-validation 10 phần chia theo nhóm merchant, lặp 5 lần, trọng số mỗi nhóm bằng nhau.

| Độ tin cậy ảnh | Brier trước → sau | AUROC trước → sau | ECE trước → sau |
|---|---:|---:|---:|
| Thô (min độ tin cậy luật) | 0,212 → 0,214 | 0,721 → 0,730 | 0,144 → 0,120 |
| Hiệu chỉnh, min (dùng) | 0,219 → 0,188 | 0,760 → 0,826 | 0,156 → 0,153 |
| Hiệu chỉnh, product | 0,240 → 0,206 | 0,761 → 0,834 | 0,216 → 0,209 |

Brier và AUROC của bản hiệu chỉnh cải thiện sau sửa luật, nhưng ECE vẫn kém độ tin cậy thô (0,153 so với 0,120).
**Brier thô tăng nhẹ** (0,212 → 0,214), LogLoss thô cũng tăng (0,844 → 0,919): luật mới thay đặc trưng/nhãn đúng sai và độ tin cậy thô chưa được khớp lại như mô hình hiệu chỉnh.
Min các xác suất trường không được đảm bảo là xác suất ảnh đã calibration tốt; dữ liệu nhỏ/lệch và GT nhiễu vẫn là giới hạn.

**Ngưỡng chạy giữ nguyên:** `tau_high = 0,60`, `tau_low = 0,50`, `target_auto_accept_error = 0,15`; `required_fields` và công thức trong `pipeline.yaml` không đổi.
Ở ngưỡng chạy 0,60, OOF có 92/194 ảnh auto-accept (coverage có trọng số nhóm 27,4%), lỗi có trọng số **9,2%**; trước sửa là 98 ảnh và lỗi 13,1% ở cùng ngưỡng.
Mô hình mới đã lưu trong `configs/doc_confidence_calibration.json`.

Ngưỡng đề xuất trong metadata giảm từ **0,5965358865 → 0,5131595764** (mới: 101/194 ảnh; coverage trọng số 29,9%; lỗi trọng số 8,5%).
Chỉ xét đề xuất **lớn hơn tau_low** để giữ đúng thứ tự routing. Đề xuất mới có lỗi không trọng số 8,9%, CI 95% bootstrap nhóm [1,8%; 17,6%]; không trộn CI này với ước lượng có trọng số.
Không tự hạ ngưỡng chạy: 0,60 đã đạt mục tiêu 15% theo ước lượng điểm và dữ liệu chọn ngưỡng chưa độc lập.
`calibrate_confidence.py` báo riêng ngưỡng đang cấu hình và đề xuất, không tự sửa YAML.

Đường risk-coverage vẫn không đơn điệu; ngưỡng dựa trên ước lượng điểm, không phải cận trên CI. Vòng sửa này không thay mục tiêu lỗi 15% và không khẳng định đạt mục tiêu 2% của kế hoạch.
Trên val (110 ảnh có tham chiếu, **in-sample**): `auto_accept` 47 ảnh, 41 đúng (87,2%); `accept_with_warning` 6 ảnh, 4 đúng (66,7%);
`human_review` 57 ảnh, 22 đúng (38,6%). Còn 3 ảnh không có tham chiếu, cũng được chuyển review; không đưa vào các tỷ lệ đúng.

## Chưa làm

Dòng hàng (`items`), VAT/giảm giá, kiểm tra tổng khớp dòng hàng, API/DB/Supabase (M6), tiền xử lý khác `none`, chấm trên test.
Đã xử lý B1 bằng tín hiệu hình học/cấu trúc, nhưng đây vẫn là heuristic: có thể gộp sai tiêu đề hoặc bỏ hậu tố thuộc tên thật; chưa có bằng chứng độc lập về khả năng tổng quát.
B4 mới xử lý phần nhãn địa chỉ tách riêng, không phục hồi mọi dòng địa chỉ OCR đọc hỏng.
B2 (nghiêng/ghép nhãn-giá trị), B3 (chữ số ngày/giờ bị đọc sai), B5 (dòng hàng/VAT), B6 (ảnh chồng/lộn/chất lượng thấp) chưa xử lý trong vòng này.
C1–C3 cần review nhãn và dữ liệu độc lập; không sửa data/GT hay mở test để làm đẹp số đo. C4 vẫn còn bất định chọn ngưỡng và giới hạn xác suất nêu trên. Bước VLM phía sau chưa triển khai.
