"""Cấu hình dự án: biến môi trường (.env) + file YAML trong configs/.

- Settings: secret/URL theo môi trường (đọc .env ở gốc repo, biến môi trường thật được ưu tiên; docker-compose ghi đè URL nội bộ).
- PipelineConfig: luật validation, ngưỡng routing, queue (configs/pipeline.yaml), kiểm tra kiểu để sai chính tả/giá trị lỗi dừng sớm.
"""
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

Prob = Annotated[float, Field(ge=0, le=1)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model_version: str = "dev"

    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None     # chỉ backend; không bao giờ gửi xuống client
    supabase_db_url: SecretStr | None = None               # dùng Session pooler (WSL2/IPv4), không dùng host direct IPv6
    supabase_storage_bucket: str = "receipts-original"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    mlflow_tracking_uri: str = "http://localhost:5000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tolerance(_Strict):
    abs_tol_vnd: float = Field(ge=0)
    rel_tol: Prob

    def limit(self, amount: float) -> float:
        return max(self.abs_tol_vnd, self.rel_tol * abs(amount))


class ValidationRules(_Strict):
    line_item: Tolerance
    total: Tolerance
    vat_formulas: dict[str, str]
    allow_future_date: bool = False


class Routing(_Strict):
    tau_high: Prob
    tau_low: Prob
    target_auto_accept_error: Prob
    required_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Routing":
        if self.tau_low >= self.tau_high:
            raise ValueError("tau_low phải nhỏ hơn tau_high")
        return self


class Bootstrap(_Strict):
    n_resamples: int = Field(gt=0)
    confidence: Annotated[float, Field(gt=0, lt=1)]
    resample_by: Literal["merchant", "document"]


class Evaluation(_Strict):
    bootstrap: Bootstrap
    primary_test: Literal["test_unseen", "test_seen"]


class PipelineConfig(_Strict):
    queues: dict[Literal["ocr", "kie"], str]
    retry: dict[str, int]
    validation: ValidationRules
    routing: Routing
    evaluation: Evaluation


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_pipeline_config() -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml("pipeline.yaml"))
