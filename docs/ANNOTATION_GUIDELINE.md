# Hướng dẫn thu thập & gán nhãn biên lai (v0.1 — dùng cho pilot)

Tài liệu này là bản nháp để pilot 30–50 biên lai. Sau pilot, chốt lại thành v1.0 trước khi gán nhãn đại trà (kế hoạch mục 12).
Mục đánh dấu **[ĐỀ XUẤT]** là chỗ ngoài kế hoạch gốc, cần xác nhận.

## 1. Mục tiêu và phạm vi

- Chỉ **biên lai bán lẻ** (retail receipt), tiếng Việt, ảnh chụp hoặc e-receipt. Không gán nhãn hoá đơn VAT đỏ, sao kê, phiếu giao hàng.
- Mục tiêu dữ liệu (kế hoạch mục 3.1): **≥ 800 biên lai, ≥ 40 merchant/layout, tối đa ~40 biên lai/merchant**, cân bằng 4 nhóm:
  `supermarket` (siêu thị), `convenience` (cửa hàng tiện lợi), `fnb`, `ecommerce` (e-commerce / hoá đơn điện tử).
- Nhãn gán ở mức **segment** (hộp quanh một dòng hoặc một cụm chữ liền nhau). Nhãn BIO theo token do code sinh tự động khi tokenize, người gán **không** gán theo từ.

## 2. Thu thập và che thông tin cá nhân (PII)

Chỉ dùng biên lai có quyền sử dụng: của bạn, gia đình, bạn bè đã đồng ý, e-receipt có đồng ý.

Luồng thư mục (đều nằm trong `data/raw/private/`, đã bị `.gitignore`, không bao giờ commit):

```
data/raw/private/original/   ảnh gốc, giữ nguyên. Label Studio KHÔNG đọc thư mục này
data/raw/private/masked/     ảnh đã che PII. Label Studio chỉ đọc thư mục này
```

**Tên file:** `<merchant>_<yyyymmdd>_<stt>.jpg`, ví dụ `winmart_20261012_003.jpg`. `<merchant>` viết thường không dấu, nối bằng `-`.
Tên merchant dùng làm khoá để chia train/val/test theo merchant, nên viết nhất quán (cùng một chuỗi cho mọi cửa hàng của chuỗi đó, ví dụ `winmart`, không phải `winmart-q1`).

**Phải che trước khi đưa vào `masked/`:** tên khách hàng, số điện thoại, mã thành viên/loyalty, số thẻ/tài khoản, QR/barcode chứa định danh.
**Được giữ:** tên cửa hàng, địa chỉ cửa hàng, mã số thuế công khai, mã hoá đơn. Che bằng hộp đen đặc (không làm mờ — mờ có thể khôi phục).
Nếu quên che một ảnh đã nạp vào Label Studio: bật cờ `pii_unmasked`, che lại ảnh trong `masked/`, nạp lại task.

**Cách chụp:**
- Đặt biên lai phẳng trên nền tối, chụp thẳng góc, đủ cả 4 mép, chữ đọc được khi phóng to.
- Tránh chói đèn/flash. Giấy nhiệt phai mực vẫn nên chụp (đó là ca khó thật), gán `quality=poor`.
- Biên lai dài: chụp một ảnh nếu chữ vẫn rõ; nếu phải chụp nhiều ảnh thì gán cờ `multi_page_or_cut`.
- E-receipt (PDF, ảnh chụp màn hình): xuất ra PNG, giữ nguyên độ phân giải gốc.
- Một ảnh = một biên lai. Ảnh có nhiều biên lai thì chụp lại.

**Sổ thu thập** [ĐỀ XUẤT]: giữ một bảng (CSV hoặc Sheet) gồm `file, merchant, category, ngày chụp, nguồn (ai cho), đã che PII (y/n)` để đếm tiến độ theo merchant và nhóm. Đây là nơi dễ thấy sớm nếu lệch nhóm.

## 3. Cách gán nhãn

Cấu hình giao diện: `configs/label_studio/receipt_kie.xml`. Mỗi vùng gồm: **hộp** (Rectangle) + **nhãn** (nếu có) + **văn bản** (transcription).

1. Vẽ hộp ôm sát một dòng chữ hoặc một cụm chữ liền mạch. Không gộp nhiều dòng vào một hộp, trừ khi chúng là một giá trị bị ngắt dòng (xem mục 5).
2. Chọn nhãn cho hộp nếu hộp thuộc 15 nhãn dưới đây. **Hộp không thuộc nhãn nào để trống nhãn** (tương đương `O`).
3. Điền văn bản đúng của hộp. Khi có pre-annotation từ PaddleOCR, chỉ sửa chỗ sai. Giữ nguyên dấu tiếng Việt như in, không sửa chính tả, không tự chuẩn hoá.
4. Chữ không đọc được: thay bằng `?` (một `?` cho một ký tự). Cả vùng không đọc được thì vẽ hộp, văn bản `???`, không gán nhãn, đồng thời bật cờ `uncertain`.
5. Cuối cùng điền thuộc tính cả biên lai (mục 6).

**Số tiền, ngày, giờ: ghi đúng như in.** `1.234.567`, `150K`, `01-10-26` đều giữ nguyên. Việc đổi sang số/ISO do `parse_vnd()` và `parse_date()` làm sau.
Không sửa số theo suy luận kể cả khi tổng không khớp: ghi như in và bật cờ `uncertain`. Lệch số là tín hiệu cho bước validation, không phải lỗi gán nhãn.

## 4. Bảng nhãn

| Nhãn | Gán cho | Không gán / lưu ý |
|---|---|---|
| `MERCHANT_NAME` | Tên cửa hàng/thương hiệu ở đầu biên lai | Slogan, tên công ty mẹ ở dòng pháp nhân khác nếu đã có tên cửa hàng (để trống nhãn). Nếu chỉ có tên công ty thì gán tên công ty |
| `MERCHANT_ADDRESS` | Địa chỉ cửa hàng. Địa chỉ nhiều dòng thì mỗi dòng một hộp, đều gán nhãn này | Số điện thoại, website |
| `TAX_CODE` [ĐỀ XUẤT] | Mã số thuế (MST) của cửa hàng | Dòng "MST:" nếu key và số cùng hộp thì gán cả hộp |
| `INVOICE_ID` [ĐỀ XUẤT] | Số hoá đơn / số biên lai / mã giao dịch | Số bàn, số thứ tự gọi món, mã đơn của khách |
| `DATE` | Ngày giao dịch | Ngày in lại, ngày hết hạn, ngày sinh nhật khuyến mãi |
| `TIME` | Giờ giao dịch (hộp riêng) | Nếu ngày và giờ nằm chung một hộp (`01/10/2026 14:32`) thì gán nhãn `DATE` cho cả hộp và ghi vào `note`: `date+time`. Converter tách giờ bằng regex |
| `ITEM_NAME` | Tên hàng/món | Mã hàng/SKU/barcode nếu tách riêng hộp thì để trống nhãn; nếu cùng hộp với tên thì gán cả hộp |
| `ITEM_QTY` | Số lượng của dòng hàng (`2`, `x2`, `0,352`) | Đơn vị (`kg`, `cái`) đi kèm thì giữ trong hộp |
| `ITEM_PRICE` | Đơn giá | Khi dòng chỉ in một số tiền (số lượng ngầm là 1), số đó gán `ITEM_AMOUNT`, không gán `ITEM_PRICE` |
| `ITEM_AMOUNT` | Thành tiền của dòng hàng | Không phải tổng hoá đơn |
| `SUBTOTAL` | Tạm tính / tổng tiền hàng trước giảm giá, thuế, phí | |
| `DISCOUNT` | Tổng giảm giá/khuyến mãi cấp hoá đơn (số tiền) | Giảm giá riêng của một món (dòng số âm ngay dưới món): gán `DISCOUNT` và nối quan hệ với món (mục 5) |
| `SERVICE_CHARGE` [ĐỀ XUẤT] | Phí phục vụ (F&B), số tiền | Phí ship thì ghi cờ ghi chú, V1 chưa có trường riêng |
| `VAT` | **Số tiền** thuế GTGT | Dòng chỉ in phần trăm (`VAT 8%`) mà không có số tiền thì để trống nhãn. Nếu `VAT 8%: 12.000` cùng hộp thì gán cả hộp |
| `TOTAL` | Số tiền khách phải trả cuối cùng (Tổng cộng / Thành tiền / Tổng thanh toán), **sau** giảm giá, thuế, phí | Tiền khách đưa, tiền thừa trả lại, tổng bằng chữ, tổng số lượng. Có nhiều dòng "tổng" thì lấy dòng cuối, sau cùng mọi điều chỉnh |

Lưu ý với MC-OCR: nhãn gốc chỉ có 4 loại (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST) và key như "Tổng tiền:" là vùng riêng không nhãn.
Với dữ liệu tự thu, nếu PaddleOCR đã tách key và value thành hai hộp thì **chỉ gán nhãn hộp value**; nếu cùng một hộp thì gán cả hộp.

## 5. Dòng hàng (line items)

Phần này quyết định chất lượng đánh giá Row F1 ở M4.

- **Mỗi dòng hàng gồm** `ITEM_NAME` + `ITEM_QTY` + `ITEM_PRICE` + `ITEM_AMOUNT` (thiếu cái nào bỏ cái đó, theo mục `ITEM_PRICE` ở bảng trên).
- **Bố cục một dòng** (tên và số cùng hàng ngang): không cần làm gì thêm, converter gom theo toạ độ y.
- **Bố cục hai dòng** (tên hàng ở dòng trên, `số lượng × đơn giá = thành tiền` ở dòng dưới): dùng **Relation** của Label Studio, nối hộp `ITEM_NAME` với các hộp số của cùng món. Đây là ground truth để đánh giá extractor, nên bắt buộc với bố cục này.
- **Tên hàng bị ngắt thành nhiều dòng:** mỗi dòng một hộp `ITEM_NAME`, nối Relation giữa các hộp, rồi nối tới hộp số.
- **Món tặng/giá 0:** vẫn là một dòng hàng, gán đủ nhãn.
- **Dòng không phải hàng** (ghi chú món, "Không đá", điều kiện khuyến mãi): để trống nhãn.
- **Số lượng lẻ theo cân** (`0,352 kg`): giữ dấu thập phân như in.

> Pilot sẽ đo xem việc nối Relation tốn bao nhiêu thời gian. Nếu quá chậm, đổi sang cách khác (ví dụ chọn số thứ tự dòng) trước khi gán đại trà.

## 6. Thuộc tính cả biên lai

| Trường | Giá trị | Cách chọn |
|---|---|---|
| `category` (bắt buộc) | supermarket / convenience / fnb / ecommerce | Theo loại cửa hàng, không theo mặt hàng |
| `payment_method` | cash / card / ewallet / bank_transfer / other | Theo dòng "Hình thức thanh toán"; không in thì bỏ trống |
| `price_includes_vat` | included / excluded / unknown | `included` khi biên lai ghi giá đã gồm VAT hoặc `TOTAL` bằng tổng thành tiền các dòng; `excluded` khi `TOTAL` = tổng hàng + VAT; không đoán được thì `unknown`. Giá trị này dùng để chọn công thức kiểm tra tổng ở M5 |
| `quality` (bắt buộc) | good / medium / poor | `good`: đọc dễ; `medium`: vài chỗ mờ/nghiêng; `poor`: mờ, phai, nhăn, khó đọc nhiều chỗ |
| `flags` | `pii_unmasked`, `multi_page_or_cut`, `rotated_or_skewed`, `not_a_receipt`, `uncertain` | Chọn nhiều. `not_a_receipt` và `pii_unmasked` loại ảnh khỏi dataset cho đến khi sửa |

Ảnh bị xoay hoặc lật ngược hẳn: xoay file ảnh trước khi nạp thay vì gán trên ảnh xoay; nghiêng nhẹ thì vẫn gán và bật `rotated_or_skewed`.

## 7. Pilot và kiểm tra nhất quán

1. **Pilot 30–50 biên lai**, đủ 4 nhóm, ít nhất 10 merchant. Ghi lại `lead_time` mỗi biên lai (Label Studio tự lưu theo từng annotation) để ước lượng tổng công gán nhãn cho 800 biên lai.
2. Sau pilot: liệt kê các ca mơ hồ gặp phải, sửa guideline thành v1.0, rồi mới gán đại trà.
3. **Sau khoảng 1 tuần**, gán lại ngẫu nhiên ~10% số biên lai (không nhìn nhãn cũ) và so sánh hai lượt: khớp theo trường (exact match sau chuẩn hoá) và khớp hộp (IoU ≥ 0.5, cùng nhãn).
   Ngưỡng "đạt": [ĐỀ XUẤT] ≥ 95% khớp với các trường bắt buộc (`MERCHANT_NAME`, `DATE`, `TOTAL`, dòng hàng), ≥ 90% với các trường còn lại. Kế hoạch yêu cầu "kiểm tra nhất quán đạt" nhưng chưa nêu con số.
4. Sau khoảng 200 biên lai sẽ train model v0 để gán nhãn gợi ý (kế hoạch mục 3.1); lúc đó vẫn người duyệt từng ảnh. **Tập test luôn do người gán, không dùng nhãn từ model.**

## 8. Xuất dữ liệu

Export JSON từ Label Studio, rồi converter (chưa viết) chuyển sang định dạng chuẩn gồm `image_id, tokens, bbox, BIO labels` và ground truth cấp biên lai theo `configs/receipt_schema.json`.
Bbox chuẩn hoá về thang 0–1000. Tập dữ liệu sau khi chốt được version bằng DVC, tách theo merchant (không để cùng merchant ở cả train và test).

## 9. Chạy công cụ

Label Studio cài trong venv riêng (`.venv-ls`, xem `requirements-annotation.txt`).

```bash
scripts/run_label_studio.sh                      # http://localhost:8080, lần đầu tạo tài khoản
python scripts/ls_make_tasks.py --limit 50       # sinh data/annotations/ls_tasks.json từ ảnh trong masked/
LS_EMAIL=<email> LS_PASSWORD=<mật khẩu> .venv-ls/bin/python scripts/ls_setup_project.py --title pilot-50
```

Label Studio chỉ phục vụ ảnh trong `masked/`; thư mục `original/` không được đăng ký nên không truy cập được qua giao diện.

## Nhật ký thay đổi

- v0.1 (2026-10-05): bản nháp cho pilot.
