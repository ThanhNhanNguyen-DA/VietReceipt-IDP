"""Tạo các experiment MLflow đã chốt trong configs/mlflow.yaml (idempotent). Tracking URI lấy từ .env (MLFLOW_TRACKING_URI).

Chạy: python scripts/init_mlflow.py [--uri http://localhost:5000]   (docker compose up mlflow)
"""
import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_settings, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=None)
    args = ap.parse_args()

    cfg = load_yaml("mlflow.yaml")
    uri = args.uri or get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    for name, desc in cfg["experiments"].items():
        exp = client.get_experiment_by_name(name)
        if exp is None:
            client.create_experiment(name, tags={**cfg.get("default_tags", {}), "mlflow.note.content": desc})
            print("created", name)
        else:
            print("exists ", name)
    print("tracking uri:", uri)


if __name__ == "__main__":
    main()
