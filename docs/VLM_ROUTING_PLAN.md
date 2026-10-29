# Kế hoạch: VLM kiểm tra lại kết quả của luật để routing

Trạng thái: **đề xuất**, chưa triển khai (2026-10-29). Bối cảnh: [RECEIPT_FLOW.md](RECEIPT_FLOW.md), [BUGS_FOR_CODEX.md](BUGS_FOR_CODEX.md).

## Thay đổi so với kế hoạch gốc

`ai_completed_plan.docx` mục 5 ghi "VLM fallback chỉ dùng ở benchmark/V2" và mục 8 coi VLM là competitor.
Kế hoạch này đưa VLM vào đường chạy như **tín hiệu routing** (không thay luật/LayoutXLM làm bộ trích chính).
Phần benchmark VLM competitor (mục 8) được làm luôn ở bước 4, cùng dữ liệu và cùng cách chấm.

## Mục tiêu và tiêu chí xong

Vấn đề hiện tại (val, rules-v2): khoảng 50% ảnh vào `human_review`; lỗi trong nhóm `auto_accept` khoảng 9% (OOF);
8/10 lỗi tự tin là tên người bán, hiệu chỉnh không bắt được vì OCR đọc sai mà vẫn tự tin.

Xong khi, trên 194 ảnh hiệu chỉnh (OOF, CV theo nhóm merchant, có CI):
1. Ở cùng coverage, lỗi `auto_accept` giảm rõ so với không có VLM (CI của hiệu số không chứa 0), **hoặc**
2. Ở cùng mức lỗi mục tiêu, tỷ lệ `human_review` giảm rõ (hướng tới ≤ 30% của kế hoạch).
3. Luồng vẫn chạy đúng khi VLM lỗi/hết thời gian (rơi về calibrator không VLM).
4. Có số đo thời gian/ảnh và VRAM đỉnh.

Nếu không đạt 1 hoặc 2 thì ghi kết quả âm và không bật VLM mặc định.

## Thiết kế

```
pages_vietocr.jsonl ──► luật (extract_fields) ──┐
                                                ├─► so khớp từng trường ─► calibrator (thêm đặc trưng VLM) ─► route()
ảnh ──► VLM trích JSON độc lập (vlm_<model>.jsonl)┘
```

**VLM trích mù, không xem kết quả của luật.** Nếu đưa giá trị của luật vào prompt ("tên là X đúng không?"), VLM nhỏ có xu hướng đồng ý,
nên tín hiệu khớp sẽ yếu. Trích độc lập cho tín hiệu "hai nguồn khác nhau cùng ra một giá trị" và đồng thời là benchmark VLM competitor.

**Đầu ra VLM** (một dòng mỗi ảnh): `image_id`, `model`, `prompt_version`, `raw`, `parsed` = `{seller, address, date, time, total}`,
`latency_s`, `error`. JSON hỏng thì thử sửa nhẹ (bỏ ``` và text thừa); vẫn hỏng thì `parsed = null`, `error = "parse"`.

**So khớp** (`src/vlm/compare.py`), dùng lại hàm chuẩn hoá có sẵn trong `src/validation/normalize.py`:

| Trường | Khớp khi |
|---|---|
| `total` | `parse_vnd` hai bên bằng nhau |
| `date` | `parse_date` hai bên bằng nhau |
| `merchant_name` | CER không phân biệt hoa/thường ≤ 0,25 (cùng ngưỡng `CLOSE_CER`) |

Mỗi trường có ba trạng thái: `agree`, `disagree`, `vlm_missing` (VLM không trả trường hoặc lỗi).

**Kết hợp vào routing:** không thêm luật tay. Thêm hai đặc trưng `vlm_agree`, `vlm_missing` vào hồi quy logistic từng trường
trong `src/pipelines/calibration.py`. Hiệu chỉnh và chọn ngưỡng đi qua đúng quy trình hiện có
(CV 10 phần theo nhóm, lặp 5 lần). Vẫn giữ min theo trường bắt buộc.
Lưu hai mô hình: `doc_confidence_calibration.json` (không VLM, dùng khi VLM lỗi) và `doc_confidence_calibration_vlm.json`.

Bước sau, chỉ khi bước trên đạt: khi luật và VLM lệch ở `total`/`date`, thử lấy giá trị VLM. Việc này đổi bộ trích nên đo riêng.

## Chọn mô hình

Giới hạn: RTX 3060 Laptop 6 GB, chạy tuần tự sau VietOCR (tiến trình riêng nên không tranh VRAM).

| Ứng viên | Ghi chú |
|---|---|
| Qwen3-VL-4B-Instruct, 4-bit | Mặc định; mạnh về tài liệu nhưng chưa có số đo tiếng Việt |
| InternVL 4B-class, 4-bit | Đối chứng |
| Vintern-1B | Chuyên tiếng Việt, nhẹ. **Rủi ro rò rỉ:** train trên dữ liệu hoá đơn Việt, có thể gồm MC-OCR; chỉ dùng nếu chứng minh được không trùng |
| API (Gemini/Claude) | Mốc trên, chỉ gửi ảnh đã che PII (kế hoạch mục 8); chưa làm ở vòng này |

Kiểm tra rò rỉ cho mọi mô hình: so độ đúng trên ảnh train MC-OCR với val; chênh bất thường thì ghi lại.

## Quy tắc dữ liệu

- Test/test_seen vẫn khoá.
- 597 ảnh train chưa dùng chỉ thuộc **4 merchant** (g0594 296, g1091 199, g1140 67, g1121 35): chỉ dùng để chỉnh định dạng prompt và tỷ lệ parse, **không** dùng để đo chất lượng.
- **Đóng băng prompt và tham số sinh trước khi chạy trên 194 ảnh hiệu chỉnh** (val + calib-train). Sau đó không sửa prompt theo lỗi trên 194 ảnh này.
- Chọn mô hình (bước 5) dùng cùng 194 ảnh với hiệu chỉnh: ghi rõ là lạc quan nhẹ, giới hạn còn lại sẽ do test trả lời.

## Các bước

| # | Việc | File | Xong khi |
|---|---|---|---|
| 0 | ✅ 2026-10-29: commit rules-v2 + hiệu chỉnh; `outputs/ocr/val-none` đã chạy lại bằng `rules-v2` | | báo cáo val-none ghi `rules-v2` |
| 1 | Môi trường: thêm `transformers`, `accelerate`, `bitsandbytes` (+ tiện ích ảnh của mô hình) vào `.venv-kie`; smoke 1 ảnh | `requirements-kie.txt` (chỉ có trên máy, bị gitignore) | VRAM đỉnh < 6 GB, có thời gian/ảnh |
| 2 | Module VLM: backend local HF (4-bit), prompt + schema JSON, parse/sửa JSON, CLI ghi `vlm_<model>.jsonl` vào thư mục run; chặn split test | `src/vlm/{backends,prompt,parse}.py`, `scripts/vlm_extract.py` | test parse và CLI xanh |
| 3 | Chỉnh prompt trên ảnh train chưa dùng (≤ 50 ảnh, chia đều 4 merchant), rồi đóng băng `prompt_version` | `configs/vlm.yaml` | tỷ lệ JSON hợp lệ ≥ 98% |
| 4 | Benchmark VLM độc lập trên 194 ảnh: độ đúng từng trường bằng `score_page`, so với luật, CI theo nhóm | `scripts/vlm_extract.py evaluate` | bảng VLM vs luật |
| 5 | Chạy ứng viên thứ hai, chọn mô hình theo độ đúng + thời gian + VRAM | | quyết định ghi vào doc |
| 6 | So khớp + đặc trưng VLM trong calibrator; risk-coverage có/không VLM trên cùng OOF; đề xuất ngưỡng | `src/vlm/compare.py`, `src/pipelines/calibration.py`, `scripts/calibrate_confidence.py --vlm` | bảng so sánh có CI hiệu số |
| 7 | Nối vào luồng: `run_receipt_flow.py` thêm bước VLM (subprocess `.venv-kie`, cờ `--vlm`); `process_page` nhận kết quả VLM; fallback khi lỗi; `model_version` ghi tên VLM | `scripts/run_receipt_flow.py`, `src/pipelines/receipt_flow.py`, `configs/pipeline.yaml` | luồng chạy cả khi bật/tắt/VLM lỗi |
| 8 | Test hồi quy, cập nhật `RECEIPT_FLOW.md`, cấu hình `routing` | `tests/test_vlm_*.py` | pytest + ruff xanh |

## Rủi ro

- **Lỗi tương quan:** VLM và OCR cùng đọc sai một ảnh mờ, hai bên vẫn "khớp" sai. Đo riêng nhóm `quality < 0,5`.
- **GT nhiễu ở tên người bán** (C2): tín hiệu khớp tên bị giới hạn bởi chính nhãn.
- **Thời gian:** VLM 4B trên GPU laptop có thể vài giây/ảnh; kế hoạch đặt P95 ≤ 5 giây cho API. Nếu quá, chỉ gọi VLM cho ảnh không phải `human_review` chắc chắn (doc_conf rất thấp) hoặc giảm độ phân giải ảnh.
- **Dữ liệu nhỏ:** 194 ảnh/70 nhóm; CI sẽ rộng. Không kết luận khi CI của hiệu số chứa 0.
