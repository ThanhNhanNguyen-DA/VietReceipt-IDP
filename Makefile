# Lệnh tắt cho dev. `make help` để xem danh sách.
PY ?= .venv/bin/python
LS_PY ?= .venv-ls/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup setup-ls hooks test cov lint fmt schema up up-gpu down mlflow-init ls ls-tasks dvc-push dvc-pull

help: ## Liệt kê lệnh
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Tạo .venv (Python 3.11) và cài môi trường dev
	python3.11 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements-dev.txt
	@test -f .env || cp .env.example .env && echo "Đã tạo .env từ .env.example: điền secret Supabase"

setup-ls: ## Tạo .venv-ls cho Label Studio
	python3.11 -m venv .venv-ls
	.venv-ls/bin/pip install -r requirements-annotation.txt

hooks: ## Cài pre-commit hook
	.venv/bin/pre-commit install

test: ## Chạy test
	$(PY) -m pytest

cov: ## Test kèm coverage
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Kiểm tra lint
	.venv/bin/ruff check .

fmt: ## Tự sửa lỗi lint có thể sửa
	.venv/bin/ruff check . --fix

schema: ## Xuất lại configs/receipt_schema.json từ Pydantic
	$(PY) scripts/export_schema.py

up: ## Dev stack: API + Redis + MLflow
	docker compose up --build

up-gpu: ## Thêm ocr-worker + kie-worker (cần GPU)
	docker compose --profile gpu up --build

down: ## Dừng stack
	docker compose --profile gpu down

mlflow-init: ## Tạo 5 experiment MLflow (cần `make up` trước)
	$(PY) scripts/init_mlflow.py

ls: ## Chạy Label Studio (http://localhost:8080)
	scripts/run_label_studio.sh

ls-tasks: ## Sinh task Label Studio từ data/raw/private/masked (pilot 50 ảnh)
	$(PY) scripts/ls_make_tasks.py --limit 50

dvc-push: ## Đẩy dữ liệu đã version lên remote DVC
	.venv/bin/dvc push

dvc-pull: ## Kéo dữ liệu từ remote DVC
	.venv/bin/dvc pull
