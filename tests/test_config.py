import pytest
import yaml
from pydantic import ValidationError

from src.config import CONFIG_DIR, PipelineConfig, Settings, get_pipeline_config, load_yaml


def test_pipeline_config_loads_and_matches_plan():
    c = get_pipeline_config()
    # tau và target lỗi là giá trị đã hiệu chỉnh (scripts/calibrate_confidence.py), không phải 0.90/0.75/0.02 khởi điểm của kế hoạch
    assert (c.routing.tau_high, c.routing.tau_low, c.routing.target_auto_accept_error) == (0.60, 0.50, 0.15)
    assert c.validation.total.limit(1_000_000) == 1000      # max(500, 0.1% * 1M)
    assert c.validation.total.limit(100_000) == 500
    assert c.validation.line_item.limit(10_000) == 100      # max(100, 0.5% * 10k = 50)
    assert set(c.queues) == {"ocr", "kie"}


def test_routing_thresholds_must_be_ordered():
    raw = load_yaml("pipeline.yaml")
    raw["routing"]["tau_low"] = 0.95
    with pytest.raises(ValidationError, match="tau_low"):
        PipelineConfig.model_validate(raw)


def test_unknown_key_is_rejected():
    raw = load_yaml("pipeline.yaml")
    raw["routing"]["tau_hgih"] = 0.9
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_mlflow_experiments_match_plan():
    names = set(yaml.safe_load((CONFIG_DIR / "mlflow.yaml").read_text(encoding="utf-8"))["experiments"])
    assert names == {"ocr-baseline", "kie-layoutxlm", "line-items", "vlm-competitor", "end-to-end"}


def test_settings_env_overrides_and_hides_secrets(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:pw@h:5432/db")
    monkeypatch.setenv("APP_ENV", "test")
    s = Settings(_env_file=None)
    assert s.app_env == "test" and s.supabase_storage_bucket == "receipts-original"
    assert "pw" not in repr(s) and s.supabase_db_url.get_secret_value().endswith("/db")
