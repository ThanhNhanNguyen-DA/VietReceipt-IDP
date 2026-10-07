"""Tạo file task cho Label Studio từ thư mục ảnh đã che PII (mặc định data/raw/private/masked).

Ảnh được phục vụ qua Local Files Storage của Label Studio (document root = data/raw/private, xem scripts/run_label_studio.sh),
nên đường dẫn trong task là /data/local-files/?d=masked/<tên>.jpg.
Có thể gắn thêm pre-annotation (kết quả OCR) bằng cách đặt <ảnh>.json cạnh ảnh theo định dạng "predictions" của Label Studio.

Chạy: python scripts/ls_make_tasks.py [--images data/raw/private/masked] [--out data/annotations/ls_tasks.json] [--limit 50]
"""
import argparse
import json
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}
DOC_ROOT = Path("data/raw/private")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=DOC_ROOT / "masked")
    ap.add_argument("--out", type=Path, default=Path("data/annotations/ls_tasks.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = lấy hết; pilot dùng 50")
    args = ap.parse_args()

    root = DOC_ROOT.resolve()
    imgs = sorted(p for p in args.images.resolve().rglob("*") if p.suffix.lower() in EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit(f"Không có ảnh trong {args.images}. Xem docs/ANNOTATION_GUIDELINE.md mục 2 (thu thập + che PII).")

    tasks = []
    for p in imgs:
        task = {"data": {"image": f"/data/local-files/?d={p.relative_to(root).as_posix()}", "image_id": p.stem}}
        pre = p.with_suffix(".json")
        if pre.exists():
            task["predictions"] = json.loads(pre.read_text(encoding="utf-8"))
        tasks.append(task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(tasks)} task)")


if __name__ == "__main__":
    main()
