
import hashlib

from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_REPOSITORY = "sovai/earnings_surprise"
DATASET_FILENAME = "earnings_surprise.parquet"
DATASET_REVISION = "15d40bd943d249f4b94d866d92986addb53d37cc"

EXPECTED_FILE_SIZE_BYTES = 15_794_937
EXPECTED_SHA256 = (
    "f0b81c85c441e552bf7976d63dd30608"
    "9cead838b5b81238e867fae131993b67"
)

HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def calculate_sha256(file_path: Path) -> str:
    """calculates the SHA-256 checksum of a file"""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE_BYTES)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def download_dataset(output_directory: Path) -> Path:
    """downloads + verifies the source earnings dataset"""
    output_directory.mkdir(parents=True, exist_ok=True)

    downloaded_file = hf_hub_download(
        repo_id=DATASET_REPOSITORY,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(output_directory),
    )

    file_path = Path(downloaded_file)

    actual_file_size = file_path.stat().st_size

    if actual_file_size != EXPECTED_FILE_SIZE_BYTES:
        raise ValueError(
            f"Unexpected dataset size: {actual_file_size} bytes"
        )

    actual_checksum = calculate_sha256(file_path)

    if actual_checksum != EXPECTED_SHA256:
        raise ValueError(
            f"Dataset checksum mismatch: {actual_checksum}"
        )

    return file_path


def main() -> None:
    output_directory = Path("data/raw")
    dataset_path = download_dataset(output_directory)

    print(f"Dataset verified: {dataset_path}")
    print(f"Repository revision: {DATASET_REVISION}")
    print(f"SHA-256: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()