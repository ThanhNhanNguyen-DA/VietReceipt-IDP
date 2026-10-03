"""Kiểm tra tính toàn vẹn của bộ MC-OCR 2021 (data/raw/mcocr). Chỉ đọc, không sửa dữ liệu.

Chạy: python scripts/check_mcocr.py [đường_dẫn_mcocr]
"""
import ast
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

SEP = "|||"
KEY_LABELS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")
csv.field_size_limit(10**9)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(root: Path) -> int:
    problems: Counter = Counter()
    examples: dict[str, list[str]] = {}

    def bad(kind: str, who: str) -> None:
        problems[kind] += 1
        examples.setdefault(kind, [])
        if len(examples[kind]) < 3:
            examples[kind].append(who)

    # ---------- train ----------
    train = read_rows(root / "mcocr_train_df.csv")
    img_dir = root / "train_images" / "train_images"
    print(f"[train] {len(train)} dòng CSV, {len(list(img_dir.glob('*.jpg')))} ảnh trong {img_dir.relative_to(root)}")
    ids = [r["img_id"] for r in train]
    if len(set(ids)) != len(ids):
        bad("duplicate_img_id", str(len(ids) - len(set(ids))))

    labels_c, missing_c, quality, sizes = Counter(), Counter(), [], Counter()
    non_nfc = ts_ok = tot_ok = tot_n = 0
    for r in train:
        who = r["img_id"]
        p = img_dir / who
        if not p.exists():
            bad("image_missing", who)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
            sizes[(w, h)] += 1
        except Exception as e:  # noqa: BLE001
            bad("image_unreadable", f"{who}: {e}")
            continue

        texts, labels = r["anno_texts"].split(SEP), r["anno_labels"].split(SEP)
        polys = ast.literal_eval(r["anno_polygons"])
        n = int(r["anno_num"])
        if not (len(texts) == len(labels) == len(polys) == n):
            bad("length_mismatch", f"{who}: texts={len(texts)} labels={len(labels)} polys={len(polys)} num={n}")
        for pol in polys:
            if (pol["width"], pol["height"]) != (w, h):
                bad("polygon_size_vs_image", f"{who}: poly {pol['width']}x{pol['height']} vs image {w}x{h}")
                break
        for pol in polys:
            x, y, bw, bh = pol["bbox"]
            if x < 0 or y < 0 or x + bw > w + 2 or y + bh > h + 2 or bw <= 0 or bh <= 0:
                bad("bbox_out_of_bounds", who)
                break

        present = set()
        for t, lb in zip(texts, labels):
            labels_c[lb or "(none)"] += 1
            present.add(lb)
            if not t.strip():
                bad("empty_text", who)
            if unicodedata.normalize("NFC", t) != t:
                non_nfc += 1
            if lb == "TOTAL_COST" and re.search(r"\d", t):
                tot_n += 1
                tot_ok += bool(re.fullmatch(r"\D*\d[\d.,\s]*\D*", t.strip()))
            if lb == "TIMESTAMP":
                ts_ok += bool(re.search(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}", t))
        for k in KEY_LABELS:
            if k not in present:
                missing_c[k] += 1
        quality.append(float(r["anno_image_quality"]))

    orphans = {p.name for p in img_dir.glob("*.jpg")} - set(ids)
    if orphans:
        bad("orphan_train_images", ", ".join(sorted(orphans)[:3]))
    print("[train] phân bố nhãn:", dict(labels_c))
    print("[train] ảnh THIẾU nhãn:", {k: missing_c[k] for k in KEY_LABELS})
    if quality:
        q = sorted(quality)
        print(f"[train] image_quality: min={q[0]:.2f} median={q[len(q)//2]:.2f} max={q[-1]:.2f}; <0.5: {sum(x < 0.5 for x in q)}")
    print("[train] kích thước ảnh phổ biến:", sizes.most_common(3))
    print(f"[train] text không chuẩn NFC: {non_nfc}; TIMESTAMP có dạng ngày: {ts_ok}/{labels_c['TIMESTAMP']}; TOTAL_COST dạng số: {tot_ok}/{tot_n}")

    # ---------- val ----------
    val = read_rows(root / "mcocr_val_sample_df.csv")
    vdir = root / "val_images" / "val_images"
    vmiss = [r["img_id"] for r in val if not (vdir / r["img_id"]).exists()]
    placeholder = sum(r["anno_texts"] == "abc abc abc" for r in val)
    print(f"[val] {len(val)} dòng, thiếu ảnh: {len(vmiss)}; dòng chỉ là placeholder 'abc abc abc' (không có nhãn thật): {placeholder}")
    if vmiss:
        bad("val_image_missing", ", ".join(vmiss[:3]))

    # ---------- text recognition crops ----------
    for name in ("text_recognition_train_data.txt", "text_recognition_val_data.txt"):
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        base = root / "text_recognition_mcocr_data" / "text_recognition_mcocr_data"
        miss = sum(1 for ln in lines if not (base / ln.split("\t")[0]).exists())
        empty = sum(1 for ln in lines if len(ln.split("\t")) < 2 or not ln.split("\t")[1].strip())
        print(f"[crops] {name}: {len(lines)} dòng, thiếu file: {miss}, text rỗng: {empty}")
        if miss:
            bad("crop_missing", name)

    print("\n=== TỔNG KẾT ===")
    if not problems:
        print("Không phát hiện vấn đề.")
    for k, v in problems.items():
        print(f"- {k}: {v}  vd: {examples[k]}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/mcocr")))
