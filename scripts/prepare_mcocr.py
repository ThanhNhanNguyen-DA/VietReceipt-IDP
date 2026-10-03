"""Chuẩn bị MC-OCR cho huấn luyện KIE/OCR: làm sạch, xoay thẳng, chuẩn hoá nhãn, chống rò rỉ, chia split.

Chạy: python scripts/prepare_mcocr.py                      (xoay thẳng ảnh -> data/processed/mcocr_clean)
      python scripts/prepare_mcocr.py --keep-orientation    (giữ hướng gốc -> data/processed/mcocr_clean_orig)
Không sửa data/raw. Cả hai chế độ cho cùng nhóm/split; orientation_fix = phép xoay cần để ảnh thẳng.
Xem README của data card trong report.json và configs/data.yaml.
"""
import ast
import csv
import difflib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.validation.normalize import parse_date, parse_time, parse_vnd  # noqa: E402

csv.field_size_limit(10**9)
DATA = Path("data")
RAW = DATA / "raw/mcocr"
KEEP_ORIENTATION = "--keep-orientation" in sys.argv   # giữ nguyên hướng ảnh gốc (không xoay thẳng)
OUT = DATA / ("processed/mcocr_clean_orig" if KEEP_ORIENTATION else "processed/mcocr_clean")
SEP = "|||"
LABEL_FIX = {"TOTAL_TOTAL_COST": "TOTAL_COST"}
KIE_LABEL = {"SELLER": "MERCHANT_NAME", "ADDRESS": "MERCHANT_ADDRESS", "TIMESTAMP": "DATETIME", "TOTAL_COST": "TOTAL"}
SEED_RANGE, TARGET, BIG_GROUP, SEEN_FRAC = 500, (0.77, 0.10, 0.13), 40, 0.06
GENERIC_SELLER = ("tên đại lý",)


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def key(s, minlen):
    k = re.sub(r"[^\w ]", "", norm(s).lower()).strip()
    return k if len(k) >= minlen and not any(g in k for g in GENERIC_SELLER) else None


def dhash(im):
    g = im.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.tobytes())
    return sum(1 << (r * 8 + c) for r in range(8) for c in range(8) if px[r * 9 + c] > px[r * 9 + c + 1])


def ham(a, b):
    return bin(a ^ b).count("1")


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def rotate_pts(pts, W, H, cw):
    return [[H - y, x] for x, y in pts] if cw else [[y, W - x] for x, y in pts]


def aabb(pts, W, H):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0, x1, y1 = max(min(xs), 0), max(min(ys), 0), min(max(xs), W), min(max(ys), H)
    return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)], [round(x0 / W * 1000), round(y0 / H * 1000), round(x1 / W * 1000), round(y1 / H * 1000)]


def pick_rotation(regs, W, H):
    """Thử xoay CW và CCW; chọn hướng cho thứ tự từ trên xuống SELLER -> ADDRESS -> TOTAL. None nếu không đủ căn cứ."""
    def ys(cw):
        out = defaultdict(list)
        for g in regs:
            if True:   # dùng cả vùng "key" (vd "TỔNG TIỀN") để định vị tổng tiền
                cx = statistics.mean(p[0] for p in g["polygon"])
                out[g["label"]].append(cx if cw else W - cx)   # y mới = x (CW) hoặc W - x (CCW)
        return {k: statistics.mean(v) for k, v in out.items()}

    def score(cw):
        y = ys(cw)
        order = [("SELLER", "ADDRESS", 1), ("SELLER", "TOTAL_COST", 2), ("ADDRESS", "TOTAL_COST", 2)]   # tiêu đề phải nằm trên tổng tiền
        return sum(w * (1 if y[a] < y[b] else -1) for a, b, w in order if a in y and b in y)

    a, b = score(True), score(False)
    return None if a == b else ("cw" if a > b else "ccw")


def load_records():
    recs, dropped = [], []
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with open(RAW / "mcocr_train_df.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            polys = ast.literal_eval(r["anno_polygons"])
            texts, labels = r["anno_texts"].split(SEP), r["anno_labels"].split(SEP)
            if not polys or not (len(polys) == len(texts) == len(labels)):
                dropped.append(r["img_id"]); continue
            src = RAW / "train_images/train_images" / r["img_id"]
            im = Image.open(src); im.load(); im = im.convert("RGB")
            W, H = im.size
            regs = []
            for p, t, lb in zip(polys, texts, labels):
                lb = LABEL_FIX.get(lb, lb)
                if lb not in KIE_LABEL or not norm(t):
                    continue
                seg = p["segmentation"][0]
                regs.append({"label": lb, "text": norm(t), "polygon": [[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)]})
            if not regs:
                dropped.append(r["img_id"]); continue
            # --- hướng chữ: box chữ cao hơn rộng => ảnh bị xoay 90 độ
            long_ = [g for g in regs if len(g["text"]) >= 5]
            shapes = [aabb(g["polygon"], W, H)[0] for g in long_]
            tall = (sum(b[3] > b[2] for b in shapes) / len(shapes)) if shapes else 0
            rot = None
            if tall > 0.6:
                rot = pick_rotation(regs, W, H)
            rel = f"raw/mcocr/train_images/train_images/{r['img_id']}"
            hash_im = im
            if rot:
                hash_im = im.transpose(Image.Transpose.ROTATE_270 if rot == "cw" else Image.Transpose.ROTATE_90)  # hash luôn trên ảnh thẳng -> nhóm/split giống nhau ở cả 2 chế độ
                if not KEEP_ORIENTATION:
                    for g in regs:
                        g["polygon"] = rotate_pts(g["polygon"], W, H, rot == "cw")
                    im = hash_im
                    W, H = im.size
                    im.save(OUT / "images" / r["img_id"], quality=95, subsampling=0)
                    rel = f"processed/mcocr_clean/images/{r['img_id']}"
            for g in regs:
                g["bbox"], g["bbox_norm"] = aabb(g["polygon"], W, H)
                g["polygon"] = [[round(x, 1), round(y, 1)] for x, y in g["polygon"]]
                g["role"] = "value" if g["label"] in ("SELLER", "ADDRESS") or re.search(r"\d", g["text"]) else "key"
                g["kie_label"] = KIE_LABEL[g["label"]] if g["role"] == "value" else "O"
            regs.sort(key=lambda g: (g["bbox"][1], g["bbox"][0]))
            recs.append({"image_id": r["img_id"], "image_path": rel, "width": W, "height": H, "quality": float(r["anno_image_quality"]),
                         "orientation_fix": rot, "orientation_applied": bool(rot) and not KEEP_ORIENTATION, "tall_text_ratio": round(tall, 2), "regions": regs, "_hash": dhash(hash_im)})
    return recs, dropped


def doc_gt(rec):
    v = lambda lb: [g for g in rec["regions"] if g["label"] == lb and g["role"] == "value"]
    totals = {parse_vnd(g["text"]) for g in v("TOTAL_COST")} - {None}
    dates = {parse_date(g["text"]) for g in v("TIMESTAMP")} - {None}
    times = {parse_time(g["text"]) for g in v("TIMESTAMP")} - {None}
    one = lambda s: next(iter(s)) if len(s) == 1 else None
    return {"merchant_name": " ".join(g["text"] for g in v("SELLER")) or None,
            "merchant_address": " ".join(g["text"] for g in v("ADDRESS")) or None,
            "transaction_date": one(dates), "transaction_time": one(times), "total": one(totals),
            "total_ambiguous": len(totals) > 1, "date_ambiguous": len(dates) > 1}


def build_groups(recs):
    n = len(recs); dsu = DSU(n); dup_of = {}
    # 1) ảnh gần như trùng (cùng biên lai chụp lại)
    for i in range(n):
        for j in range(i):
            if ham(recs[i]["_hash"], recs[j]["_hash"]) <= 3:
                dsu.union(i, j); dup_of.setdefault(i, recs[j]["image_id"])
    # 2) cùng (ngày giờ, tổng) = cùng biên lai
    by = defaultdict(list)
    for i, r in enumerate(recs):
        g = r["gt"]
        if g["transaction_date"] and g["transaction_time"] and g["total"]:
            by[(g["transaction_date"], g["transaction_time"], g["total"], g["merchant_name"] and key(g["merchant_name"], 1))].append(i)
    for v in by.values():
        for i in v[1:]: dsu.union(v[0], i); dup_of.setdefault(i, recs[v[0]]["image_id"])
    # 3) cùng merchant/địa chỉ (fuzzy) = cùng template
    for field, minlen, thr in (("merchant_name", 6, 0.86), ("merchant_address", 12, 0.90)):
        keys = defaultdict(list)
        for i, r in enumerate(recs):
            k = r["gt"][field] and key(r["gt"][field], minlen)
            if k: keys[k].append(i)
        ks = list(keys)
        for a in range(len(ks)):
            for b in range(a):
                if difflib.SequenceMatcher(None, ks[a], ks[b]).ratio() >= thr:
                    for i in keys[ks[a]]: dsu.union(i, keys[ks[b]][0])
        for v in keys.values():
            for i in v[1:]: dsu.union(v[0], i)
    return [dsu.find(i) for i in range(n)], dup_of


def assign_splits(recs, gid):
    groups = defaultdict(list)
    for i, g in enumerate(gid): groups[g].append(i)
    N = len(recs)
    feats = lambda r: [r["gt"]["merchant_name"] is not None, r["gt"]["merchant_address"] is not None, r["gt"]["transaction_date"] is not None,
                       r["gt"]["total"] is not None, r["quality"] < 0.5, r["orientation_fix"] is not None, r["quality"]]
    allf = [statistics.mean(f[k] for f in map(feats, recs)) for k in range(7)]
    big = {g: v for g, v in groups.items() if len(v) >= BIG_GROUP}
    small = [g for g in groups if g not in big]
    best = None
    for seed in range(SEED_RANGE):
        rng = random.Random(seed); order = small[:]; rng.shuffle(order)
        sp = {"val": [], "test": [], "train": []}; need = {"val": TARGET[1] * N, "test": TARGET[2] * N}; cur = "val"
        for g in order:
            if cur == "val" and len(sp["val"]) >= need["val"]: cur = "test"
            if cur == "test" and len(sp["test"]) >= need["test"]: cur = "train"
            sp[cur] += groups[g]
        for v in big.values(): sp["train"] += v
        score = 0
        for name in ("val", "test"):
            if not sp[name]: score += 9; continue
            for k in range(7):
                score += abs(statistics.mean(feats(recs[i])[k] for i in sp[name]) - allf[k]) / (allf[k] or 1)
            score += abs(len(sp[name]) - need[name]) / N * 10
        if best is None or score < best[0]: best = (score, seed, sp)
    _, seed, sp = best
    split = {}
    for name, idx in sp.items():
        for i in idx: split[i] = name
    rng = random.Random(seed)
    for v in big.values():        # test_seen: giữ lại ~6% mỗi merchant/template lớn
        cand = [i for i in v if i not in {j for j in range(N) if recs[j].get("dup_of")}]
        for i in rng.sample(cand, math.ceil(SEEN_FRAC * len(v))): split[i] = "test_seen"
    return split, seed, best[0], {g: len(v) for g, v in groups.items()}


def main():
    recs, dropped = load_records()
    if KEEP_ORIENTATION:   # dùng lại nhóm/split/gt của bản xoay thẳng để hai bản so sánh được với nhau
        ref_path = DATA / "processed/mcocr_clean/all.jsonl"
        if not ref_path.exists():
            sys.exit("Chạy chế độ mặc định trước để có split tham chiếu (data/processed/mcocr_clean/all.jsonl).")
        ref = {json.loads(l)["image_id"]: json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines()}
        for r in recs:
            r.update({k: v for k, v in ref[r["image_id"]].items() if k in ("gt", "dup_of", "group_id", "group_size", "split", "low_quality", "complete", "sample_weight")})
        seed = score = None; gcount = Counter(r["group_id"] for r in recs)
    else:
        for r in recs: r["gt"] = doc_gt(r)
        gid, dup_of = build_groups(recs)
        for i, r in enumerate(recs): r["dup_of"] = dup_of.get(i)
        split, seed, score, gsize = assign_splits(recs, gid)
        gcount = Counter(gid)
        for i, r in enumerate(recs):
            r["group_id"] = f"g{gid[i]:04d}"; r["group_size"] = gcount[gid[i]]
            r["split"] = "dropped_dup" if (r["dup_of"] and split[i] in ("val", "test", "test_seen")) else split[i]
            r["low_quality"] = r["quality"] < 0.5
            r["complete"] = all(r["gt"][k] is not None for k in ("merchant_name", "transaction_date", "total"))
        # trọng số lấy mẫu train: ~1/sqrt(kích thước nhóm), chuẩn hoá trung bình 1, chặn trên 3
        tr = [r for r in recs if r["split"] == "train"]
        tcount = Counter(r["group_id"] for r in tr)
        raw_w = [1 / math.sqrt(tcount[r["group_id"]]) for r in tr]
        mean_w = statistics.mean(raw_w)
        for r, w in zip(tr, raw_w): r["sample_weight"] = round(min(w / mean_w, 3.0), 4)

    for r in recs: r.pop("_hash")
    OUT.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for r in recs: by_split[r["split"]].append(r)
    with open(OUT / "all.jsonl", "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rs in by_split.items():
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- crops nhận dạng: chia lại theo split của ảnh, lọc nhiễu
    img_split = {r["image_id"]: r["split"] for r in recs}
    crops, rej = defaultdict(list), Counter()
    cbase = RAW / "text_recognition_mcocr_data/text_recognition_mcocr_data"
    seen = set()
    for fn in ("text_recognition_train_data.txt", "text_recognition_val_data.txt"):
        for ln in (RAW / fn).read_text(encoding="utf-8").splitlines():
            name, _, text = ln.partition("\t"); text = norm(text)
            if name in seen: rej["duplicate_line"] += 1; continue
            seen.add(name)
            sp = img_split.get(name.rsplit("_", 1)[0] + ".jpg")
            if sp is None: rej["image_dropped"] += 1; continue
            if sp == "dropped_dup": rej["dup_image"] += 1; continue
            if not text or len(text) > 100: rej["bad_text_len"] += 1; continue
            with Image.open(cbase / name) as im: w, h = im.size
            if h < 8 or w < 8 or w / h > 60 or w / h < 0.3: rej["bad_crop_shape"] += 1; continue
            crops[sp].append((f"raw/mcocr/text_recognition_mcocr_data/text_recognition_mcocr_data/{name}", text))
    for sp, rows in crops.items():
        (OUT / f"recognition_{sp}.tsv").write_text("".join(f"{p}\t{t}\n" for p, t in rows), encoding="utf-8")
    charset = sorted({c for rows in crops.values() for _, t in rows for c in t})
    (OUT / "charset.txt").write_text("".join(charset), encoding="utf-8")

    # ---- báo cáo
    def stats(rs):
        lab = Counter(g["kie_label"] for r in rs for g in r["regions"])
        return {"images": len(rs), "groups": len({r["group_id"] for r in rs}), "low_quality": sum(r["low_quality"] for r in rs),
                "rotated_fixed": sum(bool(r["orientation_fix"]) for r in rs), "complete_gt": sum(r["complete"] for r in rs),
                "mean_quality": round(statistics.mean(r["quality"] for r in rs), 3) if rs else None, "regions_by_kie_label": dict(lab)}
    rep = {"split_seed": seed, "split_score": None if score is None else round(score, 4), "dropped_images": dropped,
           "orientation_fixed": sum(bool(r["orientation_fix"]) for r in recs), "orientation_applied": not KEEP_ORIENTATION,
           "orientation_unresolved": sum(1 for r in recs if r["tall_text_ratio"] > 0.6 and not r["orientation_fix"]),
           "landscape_remaining": sum(r["width"] > r["height"] for r in recs),
           "near_duplicates": sum(bool(r["dup_of"]) for r in recs), "groups": len(gcount),
           "largest_groups": sorted(gcount.values(), reverse=True)[:8],
           "total_ambiguous": sum(r["gt"]["total_ambiguous"] for r in recs),
           "no_total": sum(r["gt"]["total"] is None for r in recs), "no_date": sum(r["gt"]["transaction_date"] is None for r in recs),
           "splits": {k: stats(v) for k, v in sorted(by_split.items())},
           "recognition_crops": {k: len(v) for k, v in crops.items()}, "recognition_rejected": dict(rej), "charset_size": len(charset)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
