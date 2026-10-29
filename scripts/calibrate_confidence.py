"""Hiệu chỉnh doc_confidence và chọn ngưỡng routing (kế hoạch mục 5), chỉ dùng các run train/val; test không được chạm tới.

  .venv/bin/python scripts/calibrate_confidence.py --runs outputs/ocr/val-none outputs/flow/calib-train --out configs/doc_confidence_calibration.json

Quy trình: trích trường không hiệu chỉnh -> gom mẫu (đặc trưng + đúng/sai từng trường) -> cross-validation theo nhóm merchant cho dự đoán
out-of-fold -> so độ tin cậy thô với bản hiệu chỉnh (min/product) -> risk-coverage -> chọn tau_high -> khớp mô hình cuối trên toàn bộ mẫu.
Mọi con số chất lượng báo cáo ở đây là out-of-fold; mô hình cuối khớp trên cùng dữ liệu nên chạy lại luồng trên các run này là in-sample.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_pipeline_config  # noqa: E402
from src.evaluation.bootstrap import ratio_ci  # noqa: E402
from src.pipelines.calibration import (  # noqa: E402
    FIELDS,
    LABEL_KEY,
    calibration_metrics,
    choose_tau_high,
    fit_calibrator,
    group_weights,
    grouped_oof,
    risk_coverage,
)
from src.pipelines.receipt_flow import MODEL_VERSION, calibration_samples, extract_run, required_keys  # noqa: E402

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="Thư mục run đã OCR (pages_vietocr.jsonl + dataset.json), train/val")
    p.add_argument("--out", type=Path, default=ROOT / "configs/doc_confidence_calibration.json")
    p.add_argument("--target-error", type=float, help="mặc định: routing.target_auto_accept_error trong configs/pipeline.yaml")
    p.add_argument("--min-docs", type=int, default=15, help="số ảnh tối thiểu trong nhóm auto-accept khi chọn tau_high")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--l2", type=float, default=1.0)
    args = p.parse_args()
    cfg = get_pipeline_config()
    target = args.target_error if args.target_error is not None else cfg.routing.target_auto_accept_error
    required = required_keys(cfg)                  # trường bắt buộc theo configs/pipeline.yaml
    req_idx = [FIELDS.index(k) for k in required]

    samples = []
    for run in args.runs:
        results = extract_run(run, calibrate=False, write=False)  # đặc trưng thô trong bộ nhớ, không ghi đè extraction đã hiệu chỉnh
        samples += calibration_samples(run, results=results)
    # ảnh "đúng" = mọi trường bắt buộc có tham chiếu đều đúng; bỏ ảnh không có tham chiếu nào cho các trường này
    samples = [s for s in samples if any(s["labels"][LABEL_KEY[k]] is not None for k in required)]
    groups = [s["group_id"] for s in samples]
    w = group_weights(groups)
    y = np.array([all(s["labels"][LABEL_KEY[k]] in (1, None) for k in required) for s in samples], dtype=float)
    issue = np.array([bool(s["validation_issues"]) for s in samples])
    print(f"Trường bắt buộc: {', '.join(required)}; mục tiêu lỗi auto_accept <= {100 * target:.1f}%")
    print(f"{len(samples)} ảnh, {len(set(groups))} nhóm merchant; ảnh đúng mọi trường bắt buộc: {100 * y.mean():.1f}% (có trọng số nhóm: {100 * np.average(y, weights=w):.1f}%)")

    oof = grouped_oof(samples, args.folds, args.repeats, l2=args.l2)
    present = np.array([[s["fields"][n]["value"] is not None for n in FIELDS] for s in samples])
    oof = np.where(present, oof, 0.0)
    raw = np.array([min(s["fields"][n]["confidence"] for n in required) for s in samples])
    oof = oof[:, req_idx]
    variants = {"raw (min độ tin cậy luật)": raw, "hiệu chỉnh, min": oof.min(1), "hiệu chỉnh, product": oof.prod(1)}
    print("\nChất lượng xác suất 'mọi trường bắt buộc đúng' (out-of-fold theo nhóm; trọng số nhóm):")
    print(f"  {'':<28}{'Brier':>8}{'LogLoss':>9}{'AUROC':>8}{'ECE':>8}")
    metrics = {}
    for name, v in variants.items():
        m = calibration_metrics(v, y, w)
        metrics[name] = m
        print(f"  {name:<28}{m['brier']:>8.3f}{m['log_loss']:>9.3f}{m['auroc']:>8.3f}{m['ece']:>8.3f}")
    conf = oof.min(1)                              # kế hoạch mục 5: doc_conf = min theo trường bắt buộc

    print("\nRisk-coverage out-of-fold (doc_confidence hiệu chỉnh, lỗi = ít nhất một trường sai; trọng số nhóm):")
    rc = risk_coverage(np.where(issue, 0.0, conf), y, w, TAUS)
    for r in rc:
        err = "-" if r["error_rate"] is None else f"{100 * r['error_rate']:.1f}%"
        print(f"  tau >= {r['tau']:.2f}  {r['documents']:>4} ảnh  coverage {100 * r['coverage']:5.1f}%  lỗi {err}")

    tau_high = choose_tau_high(np.where(issue, 0.0, conf), y, w, target, args.min_docs, tau_low=cfg.routing.tau_low)
    configured = risk_coverage(np.where(issue, 0.0, conf), y, w, [cfg.routing.tau_high])[0]
    result = {"target_error": target, "min_docs": args.min_docs, "tau_high": tau_high, "configured_tau_high": cfg.routing.tau_high,
              "configured_risk_coverage": configured}
    print(f"\nNgưỡng đang cấu hình tau_high = {cfg.routing.tau_high:.3f}; đề xuất bên dưới phải > tau_low = {cfg.routing.tau_low:.2f}.")
    if tau_high is None:
        print(f"\nKhông có ngưỡng nào đạt lỗi <= {100 * target:.0f}% với >= {args.min_docs} ảnh: auto_accept chưa dùng được.")
    else:
        sel = (conf >= tau_high) & ~issue
        ci = ratio_ci([g for g, m in zip(groups, sel, strict=True) if m], [1 - v for v, m in zip(y, sel, strict=True) if m], [1] * int(sel.sum()), 1000, 0.95, 42)
        print(f"\ntau_high = {tau_high:.3f}: {int(sel.sum())} ảnh ({100 * sel.mean():.1f}%), lỗi {100 * ci['value']:.1f}% CI95 [{100 * ci['low']:.1f}, {100 * ci['high']:.1f}] "
              f"(theo nhóm, không trọng số)")
        result["auto_accept_error_ci95"] = [ci["value"], ci["low"], ci["high"]]
    sel_warn = (conf >= cfg.routing.tau_low) & ~issue & ((tau_high is None) | (conf < (tau_high or 2)))
    review = (conf < cfg.routing.tau_low) | issue
    review_error = f"{100 * (1 - y[review].mean()):.1f}%" if review.any() else "-"
    print(f"tau_low = {cfg.routing.tau_low:.2f}: {int(sel_warn.sum())} ảnh accept_with_warning; "
          f"{int(review.sum())} ảnh human_review, trong đó {review_error} có lỗi")

    final = fit_calibrator(samples, l2=args.l2)
    final.meta.update(fit_date=dt.date.today().isoformat(), model_version=MODEL_VERSION, runs=[r.name for r in args.runs], run_paths=[str(r.resolve()) for r in args.runs],
                      tau_low=cfg.routing.tau_low, oof_metrics=metrics, risk_coverage=rc,
                      folds=args.folds, repeats=args.repeats, **result)
    args.out.write_text(json.dumps(final.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
