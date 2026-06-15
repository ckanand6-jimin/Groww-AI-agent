"""File-based run lock — prevents concurrent runs for (product, iso_week).

Mechanism:
  - On acquire: creates a ``.lock`` file containing PID + timestamp.
  - On release: deletes the ``.lock`` file.
  - On collision: if ``.lock`` exists, reads the PID; if the process is
    still alive, raises ``LockError``; if the process is dead (stale lock),
    removes the file and proceeds.

No external dependencies — uses only ``os.getpid()``, ``pathlib``, and
``os.kill(pid, 0)`` for liveness checks.
"""

from __future__ import annotations

import logging
import os
import signal
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LockError(Exception):
    """Raised when a concurrent run is already in progress."""


class RunLock:
    """File-based lock preventing concurrent runs for a single run directory.

    Args:
        run_dir: The directory where the ``.lock`` file is placed
                 (typically ``data/runs/{product}/{iso_week}/``).

    Usage::

        with RunLock(run_dir):
            ...  # run pipeline
    """

    def __init__(self, run_dir: Path) -> None:
        self.lock_path: Path = run_dir / ".lock"
        self._acquired: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the lock, cleaning up stale locks if necessary."""
        if self.lock_path.exists():
            self._check_stale()

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(
            f"{os.getpid()}|{datetime.now().isoformat()}",
            encoding="utf-8",
        )
        self._acquired = True
        logger.debug("Lock acquired: %s", self.lock_path)

    def release(self) -> None:
        """Release the lock if it was acquired by this process."""
        if not self._acquired:
            return
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
                logger.debug("Lock released: %s", self.lock_path)
            except OSError as exc:
                logger.warning("Failed to remove lock file: %s", exc)
        self._acquired = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.release()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_stale(self) -> None:
        """Check if an existing lock belongs to a dead process.

        If the locking process is still alive, raise ``LockError``.
        If the process is dead (stale lock), remove the file and log a warning.
        """
        try:
            content = self.lock_path.read_text(encoding="utf-8").strip()
            pid_str = content.split("|")[0]
            pid = int(pid_str)
        except (ValueError, IndexError, OSError):
            # Corrupted lock file — treat as stale
            logger.warning("Corrupt lock file, removing: %s", self.lock_path)
            self.lock_path.unlink(missing_ok=True)
            return

        if _is_process_alive(pid):
            raise LockError(
                f"Run already in progress (PID {pid}). "
                f"Lock file: {self.lock_path}"
            )

        # Process is dead — stale lock
        logger.warning(
            "Stale lock (PID %d no longer running), removing: %s",
            pid, self.lock_path,
        )
        self.lock_path.unlink(missing_ok=True)


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running.

    Cross-platform: on Windows, ``os.kill`` with signal 0 may behave
    differently.  We catch all expected exceptions.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — still alive
        return True
    except (OSError, ValueError):
        # ValueError: invalid signal or PID on Windows
        return False
