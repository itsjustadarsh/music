"""Download the Kaggle Spotify Tracks Dataset via kagglehub.

Usage:
    python -m src.data.fetch_spotify

Requires KAGGLE_USERNAME and KAGGLE_KEY to be set (via .env or environment) --
see FINALSTEPS.md for where to get a Kaggle API token.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from src.config import SPOTIFY_KAGGLE_HANDLE, SPOTIFY_TRACKS_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _configure_kaggle_credentials() -> None:
    """kagglehub reads KAGGLE_USERNAME/KAGGLE_KEY from the environment, or from
    ~/.kaggle/kaggle.json. We just make sure whatever is in .env is exported."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            raise RuntimeError(
                "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY in .env, "
                "or place a kaggle.json at ~/.kaggle/kaggle.json. See FINALSTEPS.md."
            )
        log.info("Using Kaggle credentials from %s", kaggle_json)
    else:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def main() -> None:
    _configure_kaggle_credentials()
    import kagglehub  # imported lazily so credential errors surface first

    log.info("Downloading Kaggle dataset %s ...", SPOTIFY_KAGGLE_HANDLE)
    download_path = Path(kagglehub.dataset_download(SPOTIFY_KAGGLE_HANDLE))
    log.info("kagglehub cached dataset at %s", download_path)

    csv_candidates = list(download_path.glob("*.csv"))
    if not csv_candidates:
        raise RuntimeError(f"No CSV found in downloaded dataset at {download_path}")
    src_csv = csv_candidates[0]

    SPOTIFY_TRACKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, SPOTIFY_TRACKS_CSV)
    log.info("Copied %s -> %s", src_csv, SPOTIFY_TRACKS_CSV)


if __name__ == "__main__":
    main()
