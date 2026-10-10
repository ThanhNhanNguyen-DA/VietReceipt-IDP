# Luồng đầu tiên trên ảnh MC-OCR

Một luồng chạy trọn từ ảnh MC-OCR đến JSON đã chuẩn hoá, có chấm điểm theo GT. Recognizer chính là **VietOCR** (`vgg_transformer`, pretrained); detector là PaddleOCR.

```
ảnh -> tiền xử lý (none) -> Paddle det -> VietOCR rec -> trích trường (luật) -> chuẩn hoá -> kiểm tra -> routing -> JSON + báo cáo
       .venv-ocr (prepare)                .venv-kie (recognize)        .venv (src/extraction, src/pipelines)
```

## Chạy

```bash
make flow LIMIT=10 OUT=outputs/flow/val-smoke      # 10 ảnh val, ~40 giây
make flow OUT=outputs/flow/val                     # toàn bộ val (113 ảnh)
.venv/bin/python scripts/run_receipt_flow.py extract --work outputs/ocr/val-none   # chỉ trích trường trên run OCR có sẵn
```

`run` gọi `scripts/ocr_baseline.py prepare/recognize` qua subprocess ở hai môi trường riêng (Paddle và Torch xung đột CUDA), rồi trích trường ở môi trường hiện tại.
`test` bị khoá: `--split` chỉ nhận `train`/`val`.

Đầu ra trong thư mục run: `extraction_vietocr.jsonl` (một dòng mỗi ảnh), `flow_rows_vietocr.jsonl` (điểm từng ảnh), `flow_report_vietocr.json`.
Mỗi dòng của `extraction_vietocr.jsonl` có `draft` (theo `ReceiptDraft`), `fields` (giá trị, độ tin cậy, luật, dòng OCR làm bằng chứng), `doc_confidence`, `routing`, `validation_issues`, `ocr.mean_rec_score`.

## Trích trường bằng luật (`src/extraction/rules.py`)

| Trường | Luật |
|---|---|
| `total` | dòng có từ khoá 'tổng cộng/thanh toán/phải trả'; số tiền cùng dòng, cùng hàng bên phải hoặc hàng liền dưới. Loại tiền khách đưa/trả lại/giảm giá/tổng kết. Không có từ khoá thì lấy số lớn nhất (độ tin cậy thấp) |
| `date`, `time` | regex ngày (`parse_date`) trong khoảng 2000..hôm nay; ưu tiên dòng có 'ngày'. Giờ ở cùng dòng/hàng với ngày, bỏ giờ mở cửa |
| `merchant_name` | dòng giống tên cửa hàng trong 8 dòng đầu (từ thương hiệu, nhiều chữ hoa), loại dòng liên hệ/tiêu đề |
| `merchant_address` | 1-3 dòng nhiều từ khoá địa chỉ trong 12 dòng đầu, có thể nằm trước hoặc sau tên người bán; loại dòng điện thoại/MST |

Số tiền chỉ nhận khi đứng riêng (không dính mã/ngày/số âm), chấp nhận dấu cách làm phân cách hàng nghìn ('35 000'), số liền phải chia hết cho 100.
Độ tin cậy trường = độ tin cậy luật x `rec_score` của dòng chứa giá trị; `doc_confidence` = min theo `merchant.name`, `transaction.date`, `total`.
MC-OCR không có nhãn dòng hàng nên `items` không nằm trong trường bắt buộc của luồng này.
LayoutXLM (M3) sẽ thay `extract_fields` qua cùng đầu ra `dict[str, FieldResult]`, và baseline luật là mức nó phải vượt.

## Kết quả trên validation (113 ảnh, 39 nhóm; CI 95% bootstrap theo nhóm)

| Trường | Đúng | CI 95% | CER (không phân biệt hoa/thường) |
|---|---|---|---|
| Tổng tiền | 74/94 = 78,7% | [64,0; 90,7] | |
| Ngày | 70/81 = 86,4% | [77,3; 93,2] | |
| Giờ | 24/39 = 61,5% | [44,4; 80,9] | |
| Tên người bán | 62/94 = 66,0% | [47,3; 80,9] | 31,6% |
| Địa chỉ | 57/94 = 60,6% | [37,5; 76,4] | 33,9% |
| Ảnh đúng cả tổng, ngày, tên | 58/110 = 52,7% | [39,1; 65,0] | |

Đúng nghĩa là: tổng/ngày/giờ khớp một giá trị GT sau chuẩn hoá; tên/địa chỉ có CER <= 25% so với GT (tên: vùng gần nhất, địa chỉ: các vùng nối theo thứ tự đọc).
Mức trần của tổng tiền: khi biết sẵn vùng GT, VietOCR đọc đúng 84/98 (xem `docs/OCR_VALIDATION_RESULTS.md`), nên 78,7% đã gần trần OCR.

**Cách đọc các con số:**
- Luật được chỉnh sau khi xem lỗi trên validation (tập phát triển), nên số này lạc quan hơn trên dữ liệu mới. Test vẫn khoá cho lần chấm cuối.
  Mẫu train 150 ảnh thử trước đó không dùng được để phát triển vì 79% thuộc 2 merchant (tổng tiền 90% ở đó, 46,8% trên val trước khi sửa).
- Phần lớn lỗi còn lại đến từ OCR/detector ('2820' thay vì '2020', '15' thay vì '16', dòng bị tách), không phải từ luật, đúng với khoảng cách E1→E2 đã đo.
- Nhãn GT của tên người bán không nhất quán ('Giảm giá', 'Tiền trả', 'TÊN ĐẠI LÝ:' được gán SELLER) nên CER tên bị kéo lên.

## Routing chưa hiệu chỉnh

`doc_confidence` cao nhất trên val chỉ là 0,838 nên ngưỡng `tau_high = 0,90` không bao giờ đạt: không ảnh nào được `auto_accept`, 19 ảnh `accept_with_warning`, 94 ảnh `human_review`.
Độ tin cậy chỉ phân biệt yếu: ảnh có `doc_confidence >= 0,70` vẫn sai 25,8% (coverage 28%). Cần hiệu chỉnh và chọn ngưỡng bằng đường risk-coverage ở M5 (bảng đầy đủ trong `flow_report_vietocr.json`).

## Chưa làm

Dòng hàng (`items`), VAT/giảm giá, kiểm tra tổng khớp dòng hàng, API/DB/Supabase (M6), tiền xử lý khác `none`, chấm trên test.
