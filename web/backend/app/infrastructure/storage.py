import hashlib
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import aiobotocore.session
from fastapi import UploadFile

from app.core.config import settings


class StorageClient:
    def __init__(self) -> None:
        self.session = aiobotocore.session.get_session()
        self.endpoint = settings.S3_ENDPOINT
        self.bucket = settings.S3_BUCKET

    # aiobotocore >= 2.12: create_client() returns an async context manager directly,
    # not a coroutine.  No await — just enter with "async with".
    def _get_client(self):
        return self.session.create_client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )

    async def upload_bytes(
        self, data: bytes, key: str, content_type: str = "application/octet-stream"
    ) -> dict:
        hasher = hashlib.sha256()
        hasher.update(data)
        size = len(data)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size > max_bytes:
            raise ValueError(
                f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)"
            )
        async with self._get_client() as client:
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        return {"key": key, "size": size, "hash_sha256": hasher.hexdigest()}

    async def ensure_bucket(self) -> None:
        async with self._get_client() as client:
            try:
                await client.create_bucket(Bucket=self.bucket)
            except client.exceptions.BucketAlreadyOwnedByYou:
                pass

    async def copy_object(self, source_key: str, dest_key: str) -> None:
        async with self._get_client() as client:
            await client.copy_object(
                Bucket=self.bucket,
                CopySource={"Bucket": self.bucket, "Key": source_key},
                Key=dest_key,
            )

    async def upload_file(self, file: UploadFile, key: str) -> dict:
        hasher = hashlib.sha256()
        size = 0
        chunks: list[bytes] = []
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
            chunks.append(chunk)
            size += len(chunk)
            max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
            if size > max_bytes:
                raise ValueError(
                    f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)"
                )
        content = b"".join(chunks)
        async with self._get_client() as client:
            await client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=file.content_type or "application/octet-stream",
            )
        return {"key": key, "size": size, "hash_sha256": hasher.hexdigest()}

    async def download_bytes(self, key: str) -> bytes:
        async with self._get_client() as client:
            resp = await client.get_object(Bucket=self.bucket, Key=key)
            async with resp["Body"] as stream:
                return await stream.read()  # type: ignore[no-any-return]

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        async with self._get_client() as client:
            return await client.generate_presigned_url(  # type: ignore[no-any-return]
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )

    async def delete(self, key: str) -> None:
        async with self._get_client() as client:
            await client.delete_object(Bucket=self.bucket, Key=key)


@lru_cache
def get_storage() -> StorageClient:
    return StorageClient()


def build_s3_key(user_id: UUID, filename: str, file_hash: str) -> str:
    ext = Path(filename).suffix
    return f"datasets/{user_id}/{file_hash[:2]}/{file_hash}{ext}"
