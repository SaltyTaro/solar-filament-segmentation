"""Download and verify only the official Kaggle competition files."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image
from tqdm import tqdm

from solar_filament.storage import ANNOTATION_NAME, dataset_root

COMPETITION = "filament-segmentation-2026"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate():
    from dotenv import load_dotenv
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Explicit dotenv location; do not print credential values.
    checkout = Path(__file__).resolve().parents[2]
    load_dotenv(checkout / ".env")
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_manifest(api, root: Path) -> list[dict]:
    files = []
    token = None
    while True:
        response = api.competition_list_files(COMPETITION, page_size=1000, page_token=token)
        files.extend(item.to_dict() for item in response.files)
        token = response.next_page_token
        if not token:
            break
    destination = root / "data" / "manifests" / "kaggle-files.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    return files


def extract_verified_members(archive_path: Path, raw_root: Path, manifest: list[dict]) -> None:
    """Extract only known members, checking CRCs, paths, sizes, and free space."""
    expected = {row["name"]: int(row["totalBytes"]) for row in manifest}
    with zipfile.ZipFile(archive_path) as archive:
        members = {item.filename: item for item in archive.infolist() if not item.is_dir()}
        if members.keys() != expected.keys():
            missing = sorted(expected.keys() - members.keys())
            extra = sorted(members.keys() - expected.keys())
            raise ValueError(
                f"Archive differs from manifest: {len(missing)} missing, {len(extra)} unexpected"
            )
        required_space = sum(expected.values()) + 512 * 1024**2
        if shutil.disk_usage(raw_root).free < required_space:
            raise OSError("Insufficient space to extract the official dataset with a safety reserve.")
        for name in tqdm(sorted(expected), desc="Extract official files", unit="file"):
            posix_path = PurePosixPath(name)
            if (
                posix_path.is_absolute()
                or ".." in posix_path.parts
                or any(":" in part for part in posix_path.parts)
            ):
                raise ValueError(f"Unsafe archive member: {name}")
            info = members[name]
            if info.file_size != expected[name]:
                raise ValueError(f"Archive size differs from API manifest for {name}")
            target = raw_root.joinpath(*posix_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading each stream checks the ZIP CRC at EOF.
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def verify_file(root: Path, row: dict) -> dict:
    relative = PurePosixPath(row["name"])
    path = (root / "data" / "raw").joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != int(row["totalBytes"]):
        raise ValueError(f"Unexpected file size for {row['name']}: {size}")
    record = {"name": row["name"], "bytes": size, "sha256": file_sha256(path)}
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        with Image.open(path) as image:
            image.load()
            if image.size != (2048, 2048) or image.mode != "L":
                raise ValueError(f"Unexpected image shape/mode for {path.name}: {image.size}, {image.mode}")
        record["width"] = 2048
        record["height"] = 2048
        record["mode"] = "L"
    elif path.name == ANNOTATION_NAME:
        data = json.loads(path.read_text(encoding="utf-8"))
        image_ids = {item["id"] for item in data["images"]}
        annotation_ids = [item["id"] for item in data["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Duplicate annotation IDs")
        if any(annotation["image_id"] not in image_ids for annotation in data["annotations"]):
            raise ValueError("Invalid annotation image reference")
        record.update(image_records=len(data["images"]), annotation_records=len(data["annotations"]))
    return record


def verify_dataset(root: Path, manifest: list[dict], workers: int = 4) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = list(
            tqdm(
                pool.map(lambda row: verify_file(root, row), manifest),
                total=len(manifest),
                desc="Verify official data",
                unit="file",
            )
        )
    annotation = json.loads((dataset_root(root) / "train" / ANNOTATION_NAME).read_text(encoding="utf-8"))
    train_filenames = {Path(row["name"]).name for row in manifest if "/train_images/" in row["name"]}
    if train_filenames != {image["file_name"] for image in annotation["images"]}:
        raise ValueError("Annotation filenames differ from the downloaded training images")
    fingerprint_source = "\n".join(
        f"{row['name']} {row['sha256']}" for row in sorted(verified, key=lambda x: x["name"])
    )
    result = {
        "competition": COMPETITION,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(root),
        "files": verified,
        "total_files": len(verified),
        "total_bytes": sum(row["bytes"] for row in verified),
        "dataset_sha256": hashlib.sha256(fingerprint_source.encode()).hexdigest(),
        "train_images": len(train_filenames),
        "test_images": sum("/test_images/" in row["name"] for row in manifest),
        "image_records": len(annotation["images"]),
        "annotation_records": len(annotation["annotations"]),
    }
    destination = root / "data" / "manifests" / "verified-dataset.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def ingest(root: Path, verify_only: bool = False, workers: int = 4) -> dict:
    raw_root = root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data" / "manifests" / "kaggle-files.json"
    if verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        api = authenticate()
        manifest = fetch_manifest(api, root)
        complete_sizes = all(
            (raw_root / row["name"]).is_file()
            and (raw_root / row["name"]).stat().st_size == int(row["totalBytes"])
            for row in manifest
        )
        if not complete_sizes:
            print(
                f"Downloading official corpus: {len(manifest)} files, {sum(int(x['totalBytes']) for x in manifest) / 2**30:.3f} GiB",
                flush=True,
            )
            api.competition_download_files(COMPETITION, path=str(raw_root), quiet=True)
            archive = raw_root / f"{COMPETITION}.zip"
            if not archive.is_file():
                raise FileNotFoundError("Kaggle did not produce the expected competition ZIP")
            if archive.stat().st_size >= 2**32:
                raise ValueError("Archive exceeds the FAT32 file-size limit")
            extract_verified_members(archive, raw_root, manifest)
    return verify_dataset(root, manifest, workers)
