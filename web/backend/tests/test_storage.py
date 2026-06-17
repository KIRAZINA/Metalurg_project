import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.infrastructure.storage import StorageClient, build_s3_key


@pytest.fixture
def mock_s3_client():
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.create_bucket = AsyncMock()
    client.put_object = AsyncMock()
    client.get_object = AsyncMock(
        return_value={
            "Body": AsyncMock(
                __aenter__=AsyncMock(
                    return_value=AsyncMock(
                        read=AsyncMock(return_value=b"test-data")
                    )
                ),
                __aexit__=AsyncMock(return_value=None),
            )
        }
    )
    client.generate_presigned_url = AsyncMock(
        return_value="https://presigned.url/test"
    )
    client.delete_object = AsyncMock()
    client.copy_object = AsyncMock()
    client.list_buckets = AsyncMock()
    return client


@pytest.fixture
def mock_session(mock_s3_client):
    session = MagicMock()
    session.create_client = AsyncMock(return_value=mock_s3_client)
    return session


@pytest.fixture
def storage(mock_session):
    with patch(
        "app.infrastructure.storage.aiobotocore.session.get_session",
        return_value=mock_session,
    ):
        s = StorageClient()
        s.session = mock_session
        yield s


class TestStorageClient:
    async def test_ensure_bucket_creates(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        mock_s3_client.create_bucket = AsyncMock()
        await storage.ensure_bucket()
        mock_s3_client.create_bucket.assert_awaited_once_with(
            Bucket=settings.S3_BUCKET
        )

    async def test_ensure_bucket_already_exists(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        class BucketError(Exception):
            pass

        mock_s3_client.exceptions.BucketAlreadyOwnedByYou = BucketError
        mock_s3_client.create_bucket = AsyncMock(side_effect=BucketError())
        await storage.ensure_bucket()

    async def test_upload_file(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file.read = AsyncMock(side_effect=[b"test-content", b""])

        result = await storage.upload_file(file, "test/key.xlsx")

        assert result["key"] == "test/key.xlsx"
        assert result["size"] == 12
        expected_hash = hashlib.sha256(b"test-content").hexdigest()
        assert result["hash_sha256"] == expected_hash
        mock_s3_client.put_object.assert_awaited_once()

    async def test_upload_file_too_large(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        big_data = b"x" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        file = MagicMock(spec=UploadFile)
        file.read = AsyncMock(side_effect=[big_data, b""])

        with pytest.raises(ValueError, match="File too large"):
            await storage.upload_file(file, "test/big.xlsx")

    async def test_upload_bytes(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        data = b"raw-bytes-data"
        result = await storage.upload_bytes(data, "test/raw.bin")

        assert result["size"] == 14
        expected_hash = hashlib.sha256(data).hexdigest()
        assert result["hash_sha256"] == expected_hash
        mock_s3_client.put_object.assert_awaited_once_with(
            Bucket=settings.S3_BUCKET,
            Key="test/raw.bin",
            Body=data,
            ContentType="application/octet-stream",
        )

    async def test_upload_bytes_too_large(
        self, storage: StorageClient
    ) -> None:
        big_data = b"x" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="File too large"):
            await storage.upload_bytes(big_data, "test/big.bin")

    async def test_download_bytes(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        mock_s3_client.get_object = AsyncMock(
            return_value={
                "Body": AsyncMock(
                    __aenter__=AsyncMock(
                        return_value=AsyncMock(
                            read=AsyncMock(return_value=b"downloaded-data")
                        )
                    ),
                    __aexit__=AsyncMock(return_value=None),
                )
            }
        )
        data = await storage.download_bytes("test/key.xlsx")
        assert data == b"downloaded-data"
        mock_s3_client.get_object.assert_awaited_once_with(
            Bucket=settings.S3_BUCKET, Key="test/key.xlsx"
        )

    async def test_get_presigned_url(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        url = await storage.get_presigned_url("test/key.xlsx")
        assert url == "https://presigned.url/test"
        mock_s3_client.generate_presigned_url.assert_awaited_once()

    async def test_delete(
        self, storage: StorageClient, mock_s3_client
    ) -> None:
        await storage.delete("test/key.xlsx")
        mock_s3_client.delete_object.assert_awaited_once_with(
            Bucket=settings.S3_BUCKET, Key="test/key.xlsx"
        )


class TestBuildS3Key:
    def test_build_s3_key_standard(self) -> None:
        from uuid import UUID

        user_id = UUID("12345678-1234-5678-1234-567812345678")
        key = build_s3_key(user_id, "test.xlsx", "abcdef1234567890")
        assert "datasets/" in key
        assert str(user_id) in key
        assert key.endswith(".xlsx")
        assert "ab" in key

    def test_build_s3_key_preserves_extension(self) -> None:
        from uuid import UUID

        user_id = UUID("12345678-1234-5678-1234-567812345678")
        key = build_s3_key(user_id, "data.xls", "deadbeef")
        assert key.endswith(".xls")
