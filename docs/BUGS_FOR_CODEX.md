# Danh sách bug và việc tồn đọng (luồng MC-OCR đầu tiên)

Tài liệu bàn giao cho agent khác. Mỗi mục có vị trí, cách tái hiện đã chạy thật (2026-10-10), kỳ vọng và tiêu chí xong.
Bối cảnh luồng: [RECEIPT_FLOW.md](RECEIPT_FLOW.md). Quy ước dữ liệu: [DATASETS.md](DATASETS.md).

## Quy tắc bắt buộc khi sửa

- **Không chạm split `test`/`test_seen`** (khoá đến lần chấm cuối; `prepare` đã chặn nếu thiếu `--allow-test`).
- **Dữ liệu đã bị "đốt" cho phát triển luật:** val (113 ảnh) và `outputs/ocr/train-dev150-none` (150 ảnh train, 79% thuộc 2 merchant). Số đo trên hai tập này lạc quan. `outputs/flow/calib-train` (85 ảnh train, <= 12 ảnh/nhóm) là dữ liệu hiệu chỉnh, không dùng để chỉnh luật.
- **Sau khi đổi luật trích trường phải chạy lại hiệu chỉnh** và cập nhật `configs/doc_confidence_calibration.json` + ngưỡng trong `configs/pipeline.yaml`:
  `.venv/bin/python scripts/calibrate_confidence.py --runs outputs/ocr/val-none outputs/flow/calib-train`
- `.venv/bin/python -m pytest` và `.venv/bin/ruff check .` phải xanh. Mỗi bug thêm ít nhất một test hồi quy.
- `outputs/`, `models/`, `data/` bị gitignore; không commit. Chạy lại luồng: `scripts/run_receipt_flow.py extract --work outputs/ocr/val-none`.
- Số liệu hiện tại để so sánh (val, 113 ảnh, `rules-v2`, đo lại 2026-10-29): tổng tiền 75/94, ngày 71/81, giờ 24/39, tên CER 27,4%, địa chỉ CER 27,3%, ảnh đúng cả ba trường 67/110.
  Bản `rules-v1` lúc viết tài liệu này: 74/94, 70/81, 24/39, 31,6%, 33,9%, 58/110.

**Trạng thái 2026-10-29:** A1–A8 đã sửa trong `rules-v2` (đã chạy lại các đoạn tái hiện bên dưới, đều ra kết quả kỳ vọng). B1 và B4 sửa một phần.
B2, B3, B5, B6 và nhóm C còn mở. Bước VLM: xem [VLM_ROUTING_PLAN.md](VLM_ROUTING_PLAN.md).

---

## A. Bug trong code (tái hiện bằng đoạn mã ngắn)

### A1. [ĐÃ SỬA rules-v2] [Cao] Loại nhầm tên người bán vì so khớp chuỗi con
- **Vị trí:** `SELLER_EXCLUDE` trong [rules.py](../src/extraction/rules.py), dùng trong `_seller` (`any(x in key for x in SELLER_EXCLUDE)`).
- **Tái hiện:**
  ```python
  extract_fields([L("Hotel Continental"), L("Ngày: 14/08/2020")], today)["merchant_name"].value   # None
  extract_fields([L("Giò chả Bà Hai")], today)["merchant_name"].value                              # None
  extract_fields([L("Intel Store")], today)["merchant_name"].value                                 # None
  ```
  Nguyên nhân: `"tel"` khớp trong "Hotel/Intel", `"gio"` khớp trong "Giò" (sau `fold`). Các từ khoá khác (`"ngay"`, `"mst"`, `"phieu"`) có cùng rủi ro.
- **Kỳ vọng:** chỉ loại khi khớp theo ranh giới từ (như `CONTACT` đã dùng `\b`), không khớp giữa từ.
- **Xong khi:** 3 ví dụ trên trả đúng tên; test mới; không làm giảm số liệu val (chạy lại `extract`).

### A2. [ĐÃ SỬA rules-v2] [Cao] `parse_vnd` không hiểu dấu cách sau dấu phân cách
- **Vị trí:** `parse_vnd` trong [normalize.py](../src/validation/normalize.py).
- **Tái hiện:**
  ```python
  parse_vnd("55, 000")       # 0       (đúng: 55000)
  parse_vnd("55 ,000")       # 0
  parse_vnd("55. 000")       # 0
  parse_vnd("1, 234, 567")   # 567     (đúng: 1234567)
  ```
  Ảnh hưởng thật: GT `'55, 000'` (val, g0231) bị tính là lỗi dù trích đúng 55000.
- **Kỳ vọng:** gộp "dấu phân cách + khoảng trắng + đúng 3 chữ số" như đã làm cho "64 000" (`_SPACE_GROUPED`).
- **Lưu ý:** `parse_vnd("6.500.00") == 650000` mơ hồ (có thể là 6.500,00); chỉ ghi nhận, không bắt buộc sửa.

### A3. [ĐÃ SỬA rules-v2] [Cao] Số lượng bị gộp vào số tiền khi cho phép dấu cách làm phân cách
- **Vị trí:** `_AMOUNT` + `amounts_in` trong [rules.py](../src/extraction/rules.py) (nhánh `[.,  ]\d{3}`).
- **Tái hiện:** `amounts_in("Tổng số 5 116,000")` → `[5116000]` (đúng: `[116000]`; "5" là số lượng). `amounts_in("SL 2 14,000")` → `[14000]` đúng, nhưng chỉ vì "14" không đủ 3 chữ số; "SL 2 140,000" sẽ lỗi tương tự.
- **Kỳ vọng:** chỉ coi khoảng trắng là phân cách hàng nghìn khi nhóm đầu 1-3 chữ số *và* cả cụm không chứa dấu phân cách khác ở giữa (cụm hỗn hợp "5 116,000" nên tách thành 5 và 116,000).
- **Xong khi:** hai ví dụ trên đúng; test cho "35 000", "1 234 567" (vẫn gộp), "5 116,000" (tách).

### A4. [ĐÃ SỬA rules-v2] [Trung bình] Số tiền liền không chia hết cho 100 bị bỏ
- **Vị trí:** `amounts_in` (điều kiện `m[1].isdigit() and value % 100`).
- **Tái hiện:** `amounts_in("Tổng cộng 45280")` → `[]`, trong khi `amounts_in("Tổng cộng 45300")` → `[45300]`. Điều kiện này được thêm để loại mã giao dịch/điện thoại (`550301`, `18006230`).
- **Kỳ vọng:** loại mã mà không loại tiền thật, ví dụ: chấp nhận số liền khi đứng ngay sau từ khoá tổng/hoặc có ký hiệu tiền (đ, VND, vnđ) kề bên; chỉ áp quy tắc chia hết 100 cho các dòng không có từ khoá.
- **Xong khi:** "Tổng cộng 45280" ra 45280; các ca mã (`PHIẾU THANH TOÁN` + `100022 - 550301`, `Hóa đơn: HDL201005-0011`, `DT:37307551`) vẫn không ra số tiền.

### A5. [ĐÃ SỬA rules-v2] [Trung bình] Địa chỉ nhận nhầm dòng thường vì token `"so "` và các token ngắn
- **Vị trí:** `ADDRESS_TOKENS` và `_address_hits` trong [rules.py](../src/extraction/rules.py).
- **Tái hiện:** `extract_fields([L("SIEU THI AN"), L("Mua so luong lon giam gia"), L("Cam on quy khach")], today)["merchant_address"].value` → `"Mua so luong lon giam gia"`.
  Token `"so "`, `"p."`, `"q."`, `"tp "`, `"xa "` khớp giữa từ/câu thường.
- **Kỳ vọng:** khớp theo ranh giới từ; `"so "` chỉ tính khi theo sau là số (`Số 5`, `so 32`).
- **Xong khi:** ví dụ trên trả `None`; CER địa chỉ trên val không tệ đi (hiện 33,9%).

### A6. [ĐÃ SỬA rules-v2] [Thấp] Loại `giam` quá rộng trong tổng tiền
- **Vị trí:** `TOTAL_HARD_EXCLUDE` (`"giam"`, `"tem"`, `"diem"`, `"vat"` trong `TOTAL_SOFT_EXCLUDE`) — cùng lớp lỗi chuỗi con như A1.
- **Tái hiện:** dòng `Tổng cộng giám đốc 50,000` (contrived) bị loại vì `giam` ⊂ `giám`; `"tem"` ⊂ `item`, `"vat"` ⊂ các từ khác.
- **Kỳ vọng:** dùng ranh giới từ; giữ nguyên `TOTAL_HARD_EXCLUDE` cho "tổng tiền đã giảm" (VietOCR đọc "ĐÃ" thành "Ã").

### A7. [ĐÃ SỬA rules-v2] [Thấp] `parse_date` đọc sai ngày dính liền
- **Vị trí:** `parse_date` / `_DATE` trong [normalize.py](../src/validation/normalize.py).
- **Tái hiện:** `parse_date("Ngày 18/03/10/2020")` → `2010-03-18` (lấy "18/03/10"), trong khi năm đầy đủ 2020 nằm ngay sau. Ưu tiên năm 4 chữ số khi có thể.

### A8. [ĐÃ SỬA rules-v2] [Thấp] `scripts/calibrate_confidence.py`
- **Hard-code `0.5` thay vì `cfg.routing.tau_low`** (dòng ~96-98): `tau_low` trong `configs/pipeline.yaml` đổi thì báo cáo của script vẫn tính theo 0,5.
- **Ghi đè `extraction_vietocr.jsonl` trong thư mục run** (`extract_run(run, calibrate=False)`, dòng ~54): chạy script xong, file trong run là bản *không* hiệu chỉnh; lần `run_receipt_flow.py extract` sau mới ghi bản hiệu chỉnh. Cần ghi sang tên khác hoặc ghi rõ.
- **Không in cảnh báo in-sample:** `scripts/run_receipt_flow.py extract` trên val/calib-train dùng mô hình đã khớp trên chính các ảnh đó nhưng báo cáo không nói rõ. Nên in "in-sample" khi run nằm trong `meta.runs` của file hiệu chỉnh.

---

## B. Lỗi hành vi của bộ trích trường (cần thiết kế, không phải một dòng sửa)

Thứ tự theo tác động trên số liệu. Mọi mục cần đo lại trên val (và không tối ưu chỉ cho val; xem "Quy tắc").

### B1. [SỬA MỘT PHẦN rules-v2] [Cao nhất] Tên người bán: nhiều dòng và dính địa chỉ
8/10 lỗi ở nhóm `doc_confidence >= 0,7` là tên người bán, và OCR đọc tự tin nên hiệu chỉnh không bắt được.
- Tên tách 2 dòng: OCR `'THỨC'`, GT `'THỨC COFFEE'` (val g0630, 3 ảnh).
- Dòng tên dính địa chỉ/ký tự thừa: `'CO.OPSMILE - A1 KHU B KTX DAI HOC QUOC GIA'`, GT `'CO.OPSMILE'` (g1100); `'Co.OPMart HAU GIANG'`, GT `'co.opmart'` (g0545).
- Chọn nhầm dòng khi không có tiêu đề rõ: `'Trả lại khách'`, `'IRANVAYS'`, `'ching ward'`, `'Tiệm trà'`.
- **Vị trí:** `_seller` ([rules.py](../src/extraction/rules.py)); hiện chỉ chọn một dòng trong 8 dòng đầu.
- **Hướng:** gộp dòng liền kề cùng cỡ chữ/hàng gần nhau; cắt tại " - " khi phần sau giống địa chỉ; dùng chiều cao dòng (to hơn dòng thường) làm tín hiệu.
- **Xong khi:** CER tên trên val giảm đáng kể (hiện 31,6%) và tính lại hiệu chỉnh.

### B2. [Cao] Tổng tiền: biên lai bị nghiêng (giá trị lệch một hàng)
- **Ví dụ thật** (val, merchant g1091 'MINIMART ANAN'): hàng `Tổng số 5 | 116,000` (GT là 116,000), hàng `Tổng cộng: VND 150,000` (150,000 là tiền khách đưa), hàng `Tiền khách trả: 34,000`. Cột giá trị nằm cao hơn cột nhãn một hàng nên `reading_rows` ghép sai.
- **Vị trí:** `reading_rows` ([geometry.py](../src/ocr/geometry.py)) + `_total`. Hàng gom theo `y_tol = 0,5 x chiều cao dòng trung vị`, không bù độ nghiêng.
- **Số liệu liên quan** (val + calib-train, 159 ảnh có tổng): luật `next_row` đúng 68,8% (16 ảnh), `fallback:max_amount` đúng 55,6% (9 ảnh), `same_row` 86,7% (128 ảnh).
- **Hướng:** ước lượng độ nghiêng từ các box (hoặc dùng cột x của các giá trị) rồi ghép nhãn-giá trị theo cột; kiểm tra chéo nhiều ứng viên cùng giá trị (`agree`).

### B3. [Trung bình] Ngày/giờ bị OCR làm hỏng, chưa có bước sửa
- Năm bị đọc sai bị loại hoàn toàn (`'21/05/2088'`, `'21/03/2820'` → `None` vì ngoài 2000..hôm nay), ngày `'15-08-2020'` thay `'16-08-2020'` (lỗi OCR thật, không sửa được bằng luật).
- Giờ: 24/39 (61,5%) trên val. Ví dụ `'Ngay bán 103/08/2020 111115'`, `'NGÀY 10 THÔNG 11/00/5020 22/11'`.
- **Hướng:** sửa các nhầm lẫn chữ số hay gặp khi chuỗi khớp mẫu ngày (0/O, 2/8, 1/7), ưu tiên ngày xuất hiện nhiều lần trên biên lai, dùng khoảng ngày của cả bộ dữ liệu (MC-OCR chủ yếu 2020) như một prior mềm.

### B4. [SỬA MỘT PHẦN rules-v2] [Trung bình] Địa chỉ
CER 33,9% trên val. Ví dụ lỗi: lấy dòng "Mã số thuế"/"Tel" (đã loại bằng `CONTACT`, vẫn còn dòng liên hệ không có từ khoá), bỏ sót dòng đầu của địa chỉ 3 dòng (`VM+ QNH ...`), không có địa chỉ ở vị trí "Địa chỉ:" (g0015).

### B5. [Thấp] Dòng hàng, VAT, kiểm tra tổng khớp tổng dòng hàng
MC-OCR không có nhãn dòng hàng nên chưa làm (thuộc M4/M5). Không có kiểm tra chéo nào bắt được lỗi chữ số như `9.117.163` đọc thay `9.091.163`.

### B6. [Thấp] Đầu vào khó
- Ảnh có **hai biên lai chồng nhau** (`mcocr_public_145013tmibr.jpg`, val g0545): detector trả dòng của cả hai, luật chọn lẫn.
- Ảnh lộn ngược: `docs/OCR_ORIENTATION_RESULTS.md` ghi biến thể 0°/180° còn bỏ sót; luồng dùng `none`.
- Ảnh chất lượng thấp (`quality < 0,5`): E2 CER 33,8% so với 17,3% (xem `docs/OCR_VALIDATION_RESULTS.md`).

---

## C. Vấn đề dữ liệu và đánh giá (không sửa trong luật; cần quyết định/làm sạch nhãn)

### C1. Nhãn GT tổng tiền không phải số tiền
Trong val, 4/94 ảnh có GT `TOTAL_COST` mà không giá trị nào parse ra số tiền >= 1.000: `'000 66'`, `'14-08-2020 21:31'`, `'Ngày: 04/20/19/19 11:25'` và `'55, 000'` (cái cuối là bug A2). Ba cái đầu là nhãn sai, luôn bị tính là lỗi. Nên có bước kiểm tra nhãn và loại/sửa khỏi mẫu số.

### C2. Nhãn GT tên người bán không nhất quán
Ví dụ GT `SELLER`: `'Giảm giá'`, `'Tiền trả'`, `'TÊN ĐẠI LÝ:'`, `'Xuân'`, `'Thanh'`, `'VM + 1S5 Tô Hiệu'` (địa chỉ). Chỉ số "tên đúng" (CER không phân biệt hoa/thường <= 25% so với vùng GT gần nhất, `CLOSE_CER` trong [receipt_flow.py](../src/pipelines/receipt_flow.py)) vì vậy bị nhiễu và ngưỡng 25% là tuỳ ý.

### C3. Tập hiệu chỉnh nhỏ và lệch
Train 832 ảnh nhưng 94% thuộc 4 merchant; hiệu chỉnh dùng 194 ảnh / 70 nhóm (val + 85 ảnh train). Val vừa dùng để chỉnh luật vừa dùng để hiệu chỉnh nên số lạc quan. Chưa có tập độc lập; test còn khoá.

### C4. Hiệu chỉnh `doc_confidence`: giới hạn đã biết
Xem mục "Routing và hiệu chỉnh" trong [RECEIPT_FLOW.md](RECEIPT_FLOW.md). Những điểm có thể cải thiện:
- Mục tiêu lỗi `auto_accept` của kế hoạch (2%) không đạt được; tạm 15% (`configs/pipeline.yaml`) vì còn VLM kiểm soát phía sau. Out-of-fold: 15,0%, CI 95% 4,8-27,7%.
- `choose_tau_high` ([calibration.py](../src/pipelines/calibration.py)) chọn ngưỡng nhỏ nhất có **ước lượng điểm** <= mục tiêu và >= 15 ảnh; không kiểm soát bất định, ngưỡng được chọn và báo cáo trên cùng dữ liệu out-of-fold; ngưỡng chọn dùng trọng số nhóm còn CI in ra không trọng số (15,0% vs 13,1%).
- Risk-coverage không đơn điệu (13% ở >= 0,6 nhưng 22% ở >= 0,8) do lỗi tên người bán tự tin. Sửa B1 trước rồi hiệu chỉnh lại.
- Brier/ECE của bản hiệu chỉnh không tốt hơn độ tin cậy thô (0,219 so với 0,212); lợi ích là AUROC và số ảnh auto-accept ở cùng mức lỗi (95-98 so với 19 ảnh).
- Nhóm `accept_with_warning` (0,50-0,60) chỉ có 12 ảnh: băng này gần như không dùng được.
- Phương án khác đã đo: bỏ `merchant.name` khỏi `required_fields` cho lỗi 4-7% ở coverage 36-51%.

---

## D. Việc chưa làm (không phải bug)

Bước VLM kiểm soát phía sau; dòng hàng (`items`); API/DB/Supabase (M6); tiền xử lý khác `none`; chấm trên test; LayoutXLM thay `extract_fields` qua cùng đầu ra `dict[str, FieldResult]`.
