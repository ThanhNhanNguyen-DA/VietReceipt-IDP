# Nguồn dữ liệu đề xuất

Đặt dữ liệu public ngoài MC-OCR vào `data/raw/public/<tên>/` (đã nằm trong `.gitignore`). Số liệu dung lượng/license lấy từ Hugging Face API ngày 2026-10-04.
Kiểm tra lại license trước khi công bố portfolio. Lệnh tải cần `pip install -U huggingface_hub` (lệnh `hf`, bản cũ là `huggingface-cli`).

| Ưu tiên | Dataset | Dùng cho | Dung lượng | License (theo HF) |
|---|---|---|---|---|
| 1 | CORD-v2: https://huggingface.co/datasets/naver-clova-ix/cord-v2 | Giai đoạn 0 KIE: nhãn line-item (`menu.nm/cnt/unitprice/price`), subtotal, tax, total, có bbox từng từ. 800 train / 100 val / 100 test | ~2.3 GB | CC BY 4.0 |
| 2 | SROIE 2019: https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE (bản đã token hoá cho LayoutLM: https://huggingface.co/datasets/darentang/sroie) | Bổ sung layout biên lai (company, date, address, total), ưu tiên thấp hơn CORD | ~0.5 GB (31 MB bản darentang) | Tag CC BY 4.0 trên HF; nguồn gốc ICDAR chưa rõ điều khoản, xác minh |
| 3 | Open Food Facts (Việt Nam): https://vn.openfoodfacts.org/ , dump: https://world.openfoodfacts.org/data | Tên sản phẩm thật để sinh biên lai synthetic (lọc `countries_tags` = vietnam) | vài trăm MB (dump đầy đủ lớn hơn) | ODbL (ghi nguồn, share-alike) |
| 4 | Font hỗ trợ tiếng Việt cho synthetic: Google Fonts (Roboto Mono, Inconsolata, Courier Prime, Be Vietnam Pro, ...) | Render biên lai giả giấy nhiệt | nhỏ | SIL OFL |
| 5 (tuỳ chọn) | iAmHieu2012/vietnamese-ocr-dataset-aggregated: https://huggingface.co/datasets/iAmHieu2012/vietnamese-ocr-dataset-aggregated ; brianhuster/VietnameseOCRdataset: https://huggingface.co/datasets/brianhuster/VietnameseOCRdataset | Pre-train/fine-tune nhận dạng chữ Việt (VietOCR). Chưa xem nội dung, kiểm tra trước khi dùng | ~1 GB / ~125 MB | MIT / Apache-2.0 |
| bỏ qua | Voxel51/consolidated_receipt_dataset (800 mẫu, CC BY 4.0) | Có vẻ là CORD v1 định dạng FiftyOne, trùng với CORD-v2 | 2.6 GB | |
| bỏ qua | katanaml-org/invoices-donut-data-v1 | Hoá đơn tiếng Anh tổng hợp, không phải biên lai | ~1 GB | MIT |

## Lệnh tải

CORD-v2:
```bash
hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir data/raw/public/cord-v2
```

SROIE (bản token hoá, nhẹ):
```bash
hf download darentang/sroie --repo-type dataset --local-dir data/raw/public/sroie
```

Open Food Facts: tải file dump tại trang `data` ở trên rồi đặt vào `data/raw/public/openfoodfacts/`.

## Không có trên các nguồn trên
- Biên lai tiếng Việt có nhãn line-item: không tìm thấy dataset công khai; phải tự thu ~800 biên lai (kế hoạch mục 3) và gán nhãn bằng Label Studio. Đây là phần quyết định chất lượng của model.
- MC-OCR 2021: bạn đã có tại `data/raw/mcocr` (bản HF `tqhuyen/MC_OCR2021` chỉ có file CSV).
