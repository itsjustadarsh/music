"""Download and prepare FMA-small dataset.

Downloads the audio and metadata, verifies checksums, and extracts.
Expects to be run once before any training.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(os.getenv("FMA_DATA_DIR", "fma_data"))
RAW_DIR = DATA_DIR / "raw"
AUDIO_DIR = RAW_DIR / "fma_small"
METADATA_DIR = RAW_DIR / "fma_metadata"

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
FMA_METADATA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(url: str, dest: Path, expected_sha1: str) -> None:
    if dest.exists():
        print(f"  {dest.name} exists, verifying...")
        if sha1_file(dest) == expected_sha1:
            print(f"  {dest.name} checksum OK, skipping download")
            return
        print(f"  {dest.name} checksum mismatch, re-downloading")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}...")
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
    actual = sha1_file(dest)
    if actual != expected_sha1:
        raise RuntimeError(f"SHA1 mismatch for {dest.name}: {actual} != {expected_sha1}")
    print(f"  {dest.name} verified")


def extract(zip_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  {dest_dir.name} already extracted, skipping")
        return
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir.parent)
    print(f"  Extracted to {dest_dir}")


def main():
    print("=== Downloading FMA-small ===")
    audio_zip = RAW_DIR / "fma_small.zip"
    meta_zip = RAW_DIR / "fma_metadata.zip"

    download_and_verify(FMA_SMALL_URL, audio_zip, FMA_SMALL_SHA1)
    download_and_verify(FMA_METADATA_URL, meta_zip, FMA_METADATA_SHA1)

    extract(audio_zip, AUDIO_DIR)
    extract(meta_zip, METADATA_DIR)

    print("=== FMA-small ready ===")
    print(f"  Audio:    {AUDIO_DIR}")
    print(f"  Metadata: {METADATA_DIR}")


if __name__ == "__main__":
    main()
