"""Tests for product configuration loading."""

from pathlib import Path

import pytest
import yaml

from pulse.config import (
    ProductConfig,
    default_config_path,
    load_product_config,
    validate_runtime_config,
)


def test_default_config_path_points_to_groww_yaml() -> None:
    path = default_config_path()
    assert path.name == "groww.yaml"
    assert path.parent.name == "config"


def test_load_groww_yaml_template_validates() -> None:
    config = load_product_config()
    assert config.product == "groww"
    assert config.display_name == "Groww"
    assert config.review_window_weeks == 10
    assert config.analysis.max_themes == 5
    assert config.delivery.google_doc.document_title == "Weekly Review Pulse — Groww"
    assert config.delivery.email.default_mode.value == "draft"
    assert config.schedule.timezone == "Asia/Kolkata"


def test_invalid_config_raises(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("product: groww\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_product_config(bad_config)


def test_review_window_weeks_out_of_range(tmp_path: Path) -> None:
    config_data = {
        "product": "groww",
        "display_name": "Groww",
        "play_store": {"app_id": "com.example.app"},
        "review_window_weeks": 0,
        "analysis": {
            "max_themes": 5,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "max_tokens_per_run": 80000,
        },
        "delivery": {
            "google_doc": {
                "document_id": "doc-123",
                "document_title": "Weekly Review Pulse — Groww",
            },
            "email": {"stakeholders": [], "default_mode": "draft"},
        },
        "schedule": {"timezone": "Asia/Kolkata", "cron": "0 6 * * 1"},
    }
    path = tmp_path / "groww.yaml"
    path.write_text(yaml.dump(config_data), encoding="utf-8")
    with pytest.raises(Exception):
        load_product_config(path)


def test_runtime_validation_flags_placeholders(tmp_path: Path) -> None:
    """Placeholders are flagged by validate_runtime_config."""
    config_data = {
        "product": "groww",
        "display_name": "Groww",
        "play_store": {"app_id": "<groww_package_id>"},
        "review_window_weeks": 10,
        "analysis": {
            "max_themes": 5,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "gpt-4o-mini",
            "max_tokens_per_run": 80000,
        },
        "delivery": {
            "google_doc": {
                "document_id": "<google_doc_id>",
                "document_title": "Test",
            },
            "email": {"stakeholders": [], "default_mode": "draft"},
        },
        "schedule": {"timezone": "Asia/Kolkata", "cron": "0 6 * * 1"},
    }
    path = tmp_path / "groww.yaml"
    path.write_text(yaml.dump(config_data), encoding="utf-8")
    config = load_product_config(path)
    errors = validate_runtime_config(config)
    assert "play_store.app_id is still a placeholder" in errors
    assert "delivery.google_doc.document_id is still a placeholder" in errors


def test_runtime_validation_passes_with_real_values(tmp_path: Path) -> None:
    config_data = {
        "product": "groww",
        "display_name": "Groww",
        "play_store": {"app_id": "com.nextbillion.groww"},
        "review_window_weeks": 10,
        "analysis": {
            "max_themes": 5,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "max_tokens_per_run": 80000,
        },
        "delivery": {
            "google_doc": {
                "document_id": "1abcDEFghiJKL",
                "document_title": "Weekly Review Pulse — Groww",
            },
            "email": {
                "stakeholders": ["team@example.com"],
                "default_mode": "draft",
            },
        },
        "schedule": {"timezone": "Asia/Kolkata", "cron": "0 6 * * 1"},
    }
    path = tmp_path / "groww.yaml"
    path.write_text(yaml.dump(config_data), encoding="utf-8")
    config = load_product_config(path)
    assert validate_runtime_config(config) == []
