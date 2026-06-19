from redis.connection import Connection as SyncConnection
from redis.connection import ConnectionPool as SyncConnectionPool
from redis.asyncio.connection import Connection as AsyncConnection
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool


_patched_sync = False
_patched_async = False


def _patch_sync_pool():
    global _patched_sync
    if _patched_sync:
        return

    original_init = SyncConnectionPool.__init__

    def _init_with_resp2(self, **kwargs):
        if kwargs.get("protocol") is None:
            kwargs["protocol"] = 2
        original_init(self, **kwargs)

    SyncConnectionPool.__init__ = _init_with_resp2
    _patched_sync = True


def _patch_async_pool():
    global _patched_async
    if _patched_async:
        return

    original_init = AsyncConnectionPool.__init__

    def _init_with_resp2(self, **kwargs):
        if kwargs.get("protocol") is None:
            kwargs["protocol"] = 2
        original_init(self, **kwargs)

    AsyncConnectionPool.__init__ = _init_with_resp2
    _patched_async = True


def apply():
    _patch_sync_pool()
    _patch_async_pool()
