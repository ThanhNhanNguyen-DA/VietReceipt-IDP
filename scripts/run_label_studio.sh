#!/usr/bin/env bash
# Chạy Label Studio local cho annotation. Dữ liệu (SQLite + export) nằm trong data/annotations/label_studio_data (không commit).
# Lần đầu: tạo tài khoản qua giao diện http://localhost:8080 hoặc đặt LABEL_STUDIO_USERNAME / LABEL_STUDIO_PASSWORD.
# Label Studio chỉ phục vụ file nằm trong Local Files Storage đã đăng ký; scripts/ls_setup_project.py chỉ đăng ký masked/,
# nên original/ (chưa che PII) không bao giờ được phục vụ.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/private/original data/raw/private/masked data/annotations/label_studio_data

export LABEL_STUDIO_BASE_DATA_DIR="$PWD/data/annotations/label_studio_data"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD/data/raw/private"
export DISABLE_SIGNUP_WITHOUT_LINK=true   # chỉ người đã có tài khoản mới đăng nhập được
export LABEL_STUDIO_DISABLE_TELEMETRY=1
export COLLECT_ANALYTICS=false

exec .venv-ls/bin/label-studio start --host 127.0.0.1 --port "${LS_PORT:-8080}" "$@"
