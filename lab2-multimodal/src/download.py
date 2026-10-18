"""Download the two pinned Kaggle sources and verify manifest SHA-256 values."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "kaggle_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value: str) -> Path:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise ValueError(f"Unsafe manifest path: {value}")
    path = ROOT.joinpath(*relative.parts).resolve()
    if not path.is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError(f"Manifest path is outside data/raw: {value}")
    return path


def refuse_mismatch(path: Path, expected: str, force: bool) -> bool:
    if not path.exists():
        return False
    if sha256(path) == expected:
        return True
    if not force:
        raise ValueError(
            f"SHA-256 mismatch: {path}. File preserved. "
            "Use --force to replace it with a manifest-verified copy."
        )
    return False


def download_archive(source: dict, archive: Path) -> None:
    url = source["versioned_download_url"]
    parsed = urlparse(url)
    version = parse_qs(parsed.query).get("datasetVersionNumber", [])
    if parsed.scheme != "https" or version != [str(source["version"])]:
        raise ValueError("Expected an HTTPS URL with the manifest dataset version.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".download", delete=False) as handle:
            temporary = Path(handle.name)
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        size += len(chunk)
                        if size > source["archive_bytes"]:
                            raise ValueError("Downloaded archive is larger than the manifest.")
                        digest.update(chunk)
                        handle.write(chunk)
        if size != source["archive_bytes"] or digest.hexdigest() != source["archive_sha256"]:
            raise ValueError("Downloaded archive fails manifest size/SHA-256 verification.")
        temporary.replace(archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_verified(source: dict, archive: Path, target: Path) -> None:
    # The manifest permits one exact root member per archive. Never extractall.
    member_name = target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with ZipFile(archive) as bundle:
            matches = [item for item in bundle.infolist() if item.filename == member_name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one ZIP member named {member_name!r}.")
            item = matches[0]
            if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("The expected ZIP member must be a regular file.")
            if item.file_size != source["extracted_bytes"]:
                raise ValueError("ZIP member size differs from the manifest.")
            with bundle.open(item) as stream, tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".extract", delete=False
            ) as handle:
                temporary = Path(handle.name)
                shutil.copyfileobj(stream, handle, length=1024 * 1024)
        if temporary.stat().st_size != source["extracted_bytes"]:
            raise ValueError("Extracted file size differs from the manifest.")
        if sha256(temporary) != source["extracted_sha256"]:
            raise ValueError("Extracted file fails manifest SHA-256 verification.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Replace existing checksum-mismatched files after verifying new copies.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Check every existing destination before making changes to any source.
    pending = []
    for source in manifest["sources"]:
        target = local_path(source["extracted_path"])
        if refuse_mismatch(target, source["extracted_sha256"], args.force):
            print(f"Verified, unchanged: {target.relative_to(ROOT)}")
            continue
        archive = local_path(source["archive_path"])
        archive_ok = refuse_mismatch(archive, source["archive_sha256"], args.force)
        pending.append((source, archive, target, archive_ok))
    for source, archive, target, archive_ok in pending:
        if not archive_ok:
            print(f"Downloading pinned dataset: {source['ref']} v{source['version']}")
            download_archive(source, archive)
        extract_verified(source, archive, target)
        print(f"Verified: {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
