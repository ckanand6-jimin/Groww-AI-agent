"""Product configuration loading and validation."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal
from datetime import datetime, timedelta

import yaml
from pydantic import BaseModel, Field, field_validator


class EmailMode(str, Enum):
    DRAFT = "draft"
    SEND = "send"
    SKIP = "skip"


class PlayStoreConfig(BaseModel):
    app_id: str = Field(..., min_length=1)

    @field_validator("app_id")
    @classmethod
    def app_id_must_not_be_placeholder(cls, value: str) -> str:
        if value.strip().startswith("<") and value.strip().endswith(">"):
            # Placeholders are allowed at load time for template configs;
            # runtime validation can call validate_runtime_config().
            return value
        return value


class AnalysisConfig(BaseModel):
    max_themes: int = Field(default=5, ge=1, le=20)
    embedding_model: str = Field(..., min_length=1)
    llm_model: str = Field(..., min_length=1)
    max_tokens_per_run: int = Field(default=80_000, ge=1)


class GoogleDocConfig(BaseModel):
    document_id: str = Field(..., min_length=1)
    document_title: str = Field(..., min_length=1)


class EmailConfig(BaseModel):
    stakeholders: list[str] = Field(default_factory=list)
    default_mode: EmailMode = EmailMode.DRAFT


class DeliveryConfig(BaseModel):
    google_doc: GoogleDocConfig
    email: EmailConfig


class ScheduleConfig(BaseModel):
    timezone: str = Field(default="Asia/Kolkata", min_length=1)
    cron: str = Field(..., min_length=1)


class ProductConfig(BaseModel):
    """Minimal schema for config/groww.yaml."""

    product: Literal["groww"]
    display_name: str = Field(..., min_length=1)
    play_store: PlayStoreConfig
    review_window_weeks: int = Field(default=10, ge=1, le=52)
    analysis: AnalysisConfig
    delivery: DeliveryConfig
    schedule: ScheduleConfig


def default_config_path() -> Path:
    """Resolve config/groww.yaml relative to the repository root."""
    override = os.environ.get("PULSE_CONFIG_PATH")
    if override:
        return Path(override)

    # pulse-agent/src/pulse/config.py -> repo root is three parents up from pulse/
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "groww.yaml"


def load_product_config(path: Path | None = None) -> ProductConfig:
    """Load and validate groww.yaml."""
    config_path = path or default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Product config not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {config_path}")

    return ProductConfig.model_validate(raw)


def is_placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def validate_runtime_config(config: ProductConfig) -> list[str]:
    """Return human-readable errors for unset placeholder values."""
    errors: list[str] = []

    if is_placeholder(config.play_store.app_id):
        errors.append("play_store.app_id is still a placeholder")

    if is_placeholder(config.delivery.google_doc.document_id):
        errors.append("delivery.google_doc.document_id is still a placeholder")

    if config.delivery.email.default_mode == EmailMode.SEND and not config.delivery.email.stakeholders:
        errors.append("delivery.email.stakeholders must not be empty when default_mode is send")

    return errors

def get_date_window_from_iso_week(iso_week: str, window_weeks: int) -> tuple[datetime, datetime]:
    """
    Compute start_date and end_date from an iso_week string (e.g. '2026-W23').
    The end_date is the Sunday of that week.
    The start_date is `window_weeks` prior.
    """
    from datetime import datetime, timedelta
    
    # Parse the Monday of the given ISO week
    # Note: %G is ISO year, %V is ISO week, %u is ISO weekday (1=Monday)
    monday = datetime.strptime(f"{iso_week}-1", "%G-W%V-%u")
    
    # End date is the Sunday of that week at 23:59:59
    end_date = monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    # Start date is window_weeks prior to the end date (going back full weeks)
    start_date = monday - timedelta(weeks=window_weeks - 1)
    
    return start_date, end_date
