"""Upload local data folders to Azure Blob Storage.

Uses Entra ID via DefaultAzureCredential (so `az login` is the only auth step).
No account keys required.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _client(account: str) -> BlobServiceClient:
    # Silence the very chatty per-request HTTP logger from azure-core.
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


def upload_directory(
    local_dir: Path,
    container: str,
    blob_prefix: str = "",
    *,
    account: str | None = None,
    overwrite: bool = False,
) -> int:
    """Upload every file in ``local_dir`` recursively to ``container/blob_prefix/...``.

    Returns the number of blobs uploaded (skipped files do not count).
    """
    load_dotenv()
    account = account or os.environ["BLOB_ACCOUNT"]
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    container_client = _client(account).get_container_client(container)

    uploaded = 0
    for path in local_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{blob_prefix.rstrip('/')}/{rel}" if blob_prefix else rel
        blob = container_client.get_blob_client(blob_name)
        if not overwrite and blob.exists():
            continue
        with path.open("rb") as f:
            blob.upload_blob(f, overwrite=overwrite)
        uploaded += 1
        if uploaded % 100 == 0:
            logger.info("Uploaded %d blobs…", uploaded)

    logger.info("Uploaded %d blobs to %s/%s", uploaded, container, blob_prefix)
    return uploaded


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    raw_container = os.environ.get("BLOB_CONTAINER_RAW", "raw")
    splits_container = os.environ.get("BLOB_CONTAINER_SPLITS", "splits")

    raw_dir = Path("data/raw/neu")
    splits_dir = Path("data/splits")

    if raw_dir.exists():
        upload_directory(raw_dir, raw_container, blob_prefix="neu")
    if splits_dir.exists():
        upload_directory(splits_dir, splits_container, blob_prefix="neu")


if __name__ == "__main__":
    main()
