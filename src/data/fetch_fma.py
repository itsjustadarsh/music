"""Download and checksum-verify FMA-small audio + metadata archives.

Usage:
    python -m src.data.fetch_fma [--skip-audio] [--skip-metadata]

The audio archive is 7.2 GiB; the metadata archive is 342 MiB. Both are
downloaded to data/raw/ and extracted in place. Re-running is idempotent:
already-extracted archives are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import (
    FMA_METADATA_DIR,
    FMA_METADATA_ZIP_SHA1,
    FMA_METADATA_ZIP_URL,
    FMA_SMALL_AUDIO_DIR,
    FMA_SMALL_ZIP_SHA1,
    FMA_SMALL_ZIP_URL,
    RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, resuming a partial `.part` file if one is present.

    These archives are multi-GB over a link that can take hours, so an interrupted
    transfer (sleep, dropped wifi) must not mean starting from zero. We ask for a
    byte range past what we already have; servers that ignore Range respond 200
    instead of 206, in which case we start over rather than corrupt the file.
    """
    if dest.exists():
        log.info("Already downloaded: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    if resume_from:
        log.info("Resuming %s at %.1f MiB", dest.name, resume_from / 1024 / 1024)
    else:
        log.info("Downloading %s -> %s", url, dest)

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        resumed = r.status_code == 206
        if resume_from and not resumed:
            log.warning("Server ignored Range request; restarting %s from zero", dest.name)
            resume_from = 0
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resumed else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total, initial=resume_from, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)


def verify_sha1(path: Path, expected: str) -> bool:
    log.info("Verifying SHA1 for %s ...", path.name)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    actual = h.hexdigest()
    ok = actual == expected
    if not ok:
        log.error("SHA1 mismatch for %s: expected %s, got %s", path.name, expected, actual)
    else:
        log.info("SHA1 OK for %s", path.name)
    return ok


def extract(zip_path: Path, dest_dir: Path, force: bool = False) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        log.info("Already extracted: %s", dest_dir)
        return
    log.info("Extracting %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)


def fetch_one(url: str, sha1: str, zip_name: str, extracted_dir: Path, force: bool) -> None:
    zip_path = RAW_DIR / zip_name
    download(url, zip_path)
    if not verify_sha1(zip_path, sha1):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum verification failed for {zip_name}. The partial/corrupt file has been "
            "deleted; re-run this script to retry the download. If this keeps failing, FMA's "
            "hosted file may have changed -- see FINALSTEPS.md troubleshooting section."
        )
    extract(zip_path, extracted_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-audio", action="store_true", help="Skip the 7.2 GiB fma_small.zip")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip fma_metadata.zip")
    parser.add_argument("--force", action="store_true", help="Re-extract even if target dir is non-empty")
    args = parser.parse_args()

    if not args.skip_metadata:
        fetch_one(FMA_METADATA_ZIP_URL, FMA_METADATA_ZIP_SHA1, "fma_metadata.zip", FMA_METADATA_DIR, args.force)
    if not args.skip_audio:
        fetch_one(FMA_SMALL_ZIP_URL, FMA_SMALL_ZIP_SHA1, "fma_small.zip", FMA_SMALL_AUDIO_DIR, args.force)

    log.info("Done. Audio: %s | Metadata: %s", FMA_SMALL_AUDIO_DIR, FMA_METADATA_DIR)


if __name__ == "__main__":
    main()
