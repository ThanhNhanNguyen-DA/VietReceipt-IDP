"""Tạo project Label Studio cho annotation biên lai: nạp configs/label_studio/receipt_kie.xml, đăng ký Local Files Storage cho masked/
rồi import task từ ls_make_tasks.py. Label Studio phải đang chạy (scripts/run_label_studio.sh).

Đăng nhập bằng email/mật khẩu qua biến môi trường LS_EMAIL, LS_PASSWORD (Label Studio 1.23 tắt token legacy mặc định).
Chạy: python scripts/ls_setup_project.py --title "pilot-50" --tasks data/annotations/ls_tasks.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import requests

MASKED = Path("data/raw/private/masked").resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LS_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--title", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/label_studio/receipt_kie.xml"))
    ap.add_argument("--tasks", type=Path, default=Path("data/annotations/ls_tasks.json"))
    args = ap.parse_args()

    s = requests.Session()
    page = s.get(f"{args.url}/user/login")
    csrf = re.search(r'csrfmiddlewaretoken" value="([^"]+)"', page.text).group(1)
    r = s.post(f"{args.url}/user/login", headers={"Referer": args.url},
               data={"email": os.environ["LS_EMAIL"], "password": os.environ["LS_PASSWORD"], "csrfmiddlewaretoken": csrf})
    r.raise_for_status()
    h = {"X-CSRFToken": s.cookies["csrftoken"], "Referer": args.url}

    def call(method: str, path: str, **kw):
        resp = s.request(method, f"{args.url}{path}", headers=h, **kw)
        if not resp.ok:
            raise SystemExit(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")
        return resp.json() if resp.content else None

    pid = call("POST", "/api/projects/", json={"title": args.title, "label_config": args.config.read_text(encoding="utf-8")})["id"]
    call("POST", "/api/storages/localfiles", json={"project": pid, "path": str(MASKED), "title": "masked", "use_blob_urls": False})
    res = call("POST", f"/api/projects/{pid}/import", json=json.loads(args.tasks.read_text(encoding="utf-8")))
    print(f"project {pid}: {res['task_count']} task -> {args.url}/projects/{pid}/data")


if __name__ == "__main__":
    main()
