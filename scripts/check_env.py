"""Checks which credentials in .env are actually filled in (vs. still the
.env.example placeholder values) and prints a clear per-service status.

Usage:
    python scripts/check_env.py            # human-readable report, exit 0 always
    python scripts/check_env.py --require kaggle    # exit 1 if Kaggle creds missing
    python scripts/check_env.py --require spotify   # exit 1 if Spotify creds missing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

PLACEHOLDER_VALUES = {
    "your-kaggle-username", "your-kaggle-api-key",
    "your-spotify-client-id", "your-spotify-client-secret",
}


def _is_set(name: str) -> bool:
    value = os.getenv(name, "")
    return bool(value) and value not in PLACEHOLDER_VALUES


def kaggle_ok() -> bool:
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return _is_set("KAGGLE_USERNAME") and _is_set("KAGGLE_KEY")


def spotify_ok() -> bool:
    return _is_set("SPOTIFY_CLIENT_ID") and _is_set("SPOTIFY_CLIENT_SECRET") and _is_set("SPOTIFY_REDIRECT_URI")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", choices=["kaggle", "spotify"], help="Exit 1 if this service's creds are missing")
    parser.add_argument("--quiet", action="store_true", help="Suppress the report, only used with --require")
    args = parser.parse_args()

    k_ok, s_ok = kaggle_ok(), spotify_ok()

    if not args.quiet:
        print("Credential status:")
        print(f"  Kaggle  (Spotify Tracks Dataset download): {'OK' if k_ok else 'MISSING'}")
        print(f"  Spotify (OAuth / playlist creation):       {'OK' if s_ok else 'MISSING'}")
        if not k_ok:
            print("    -> Set KAGGLE_USERNAME + KAGGLE_KEY in .env, or place kaggle.json at ~/.kaggle/kaggle.json")
            print("       Get a token: kaggle.com -> profile -> Settings -> API -> Create New Token")
        if not s_ok:
            print("    -> Set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI in .env")
            print("       Get them: developer.spotify.com/dashboard -> Create app")

    if args.require == "kaggle" and not k_ok:
        sys.exit(1)
    if args.require == "spotify" and not s_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
