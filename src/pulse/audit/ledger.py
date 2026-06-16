"""Run ledger — JSON-file persistence for (product, iso_week) runs.

The ledger is the **sole source of truth** for idempotency and audit.
Each run is stored as ``data/runs/{product}/{iso_week}/run.json``.

Writes are atomic: the record is first written to a ``.tmp`` file, then
renamed to ``run.json``, preventing corruption from interrupted I/O.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from pulse.models.models import RunRecord

logger = logging.getLogger(__name__)


class LedgerError(Exception):
    """Raised on ledger read/write failures."""


def _default_runs_dir() -> Path:
    """Resolve ``data/runs/`` relative to the repository root.

    Path: ``pulse-agent/src/pulse/audit/ledger.py`` -> 4 parents up = repo root.
    Falls back to ``PULSE_DATA_DIR`` env var if set.
    """
    override = os.environ.get("PULSE_DATA_DIR")
    if override:
        return Path(override) / "runs"
    return Path(__file__).resolve().parents[4] / "data" / "runs"


class Ledger:
    """JSON-file run ledger for (product, iso_week) runs.

    Args:
        base_dir: Override the default ``data/runs/`` directory
                  (useful in tests with ``tmp_path``).
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir: Path = base_dir or _default_runs_dir()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def run_dir(self, product: str, iso_week: str) -> Path:
        return self.base_dir / product / iso_week

    def ledger_path(self, product: str, iso_week: str) -> Path:
        return self.run_dir(product, iso_week) / "run.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def load(self, product: str, iso_week: str) -> Optional[RunRecord]:
        """Load an existing RunRecord, or return ``None`` if not found."""
        path = self.ledger_path(product, iso_week)
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            return RunRecord.model_validate_json(text)
        except Exception as exc:
            raise LedgerError(
                f"Failed to load ledger at {path}: {exc}"
            ) from exc

    def save(self, record: RunRecord) -> None:
        """Persist a RunRecord to JSON.

        Creates parent directories if they do not exist.
        Uses direct write for cross-platform compatibility.
        """
        path = self.ledger_path(record.product, record.iso_week)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(
                record.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            raise LedgerError(
                f"Failed to save ledger at {path}: {exc}"
            ) from exc

        logger.debug("Ledger saved: %s", path)

    def exists(self, product: str, iso_week: str) -> bool:
        """Return True if a ledger file exists for this run."""
        return self.ledger_path(product, iso_week).is_file()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, product: str) -> List[RunRecord]:
        """List all runs for a product, sorted by iso_week descending."""
        product_dir = self.base_dir / product
        if not product_dir.is_dir():
            return []

        records: list[RunRecord] = []
        for week_dir in product_dir.iterdir():
            if not week_dir.is_dir():
                continue
            ledger_file = week_dir / "run.json"
            if not ledger_file.is_file():
                continue
            try:
                text = ledger_file.read_text(encoding="utf-8")
                records.append(RunRecord.model_validate_json(text))
            except Exception:
                logger.warning("Skipping corrupt ledger: %s", ledger_file)

        records.sort(key=lambda r: r.iso_week, reverse=True)
        return records

    def latest(self, product: str) -> Optional[RunRecord]:
        """Return the most recent completed run for a product."""
        for record in self.query(product):
            if record.is_completed():
                return record
        return None
