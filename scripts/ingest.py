"""Run document ingestion from the command line."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings_or_raise
from app.services.ingestion import run_ingestion
from app.utils.logging import logger


def main() -> int:
    """Execute the ingestion pipeline."""
    try:
        get_settings_or_raise()
        result = run_ingestion()
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Ingestion complete.")
    print(f"Files processed : {result.files_processed}")
    print(f"Pages processed : {result.pages_processed}")
    print(f"Chunks created  : {result.chunks_created}")
    print(f"Index path      : {result.index_path}")
    print(f"Chunks path     : {result.chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
