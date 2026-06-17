import prometheus_fastapi_instrumentator.routing as routing
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# Monkey-patch prometheus routing to handle _IncludedRouter objects
_original_get_route_name = routing._get_route_name


def _patched_get_route_name(scope, routes):
    filtered = [r for r in routes if hasattr(r, "path")]
    return _original_get_route_name(scope, filtered)


routing._get_route_name = _patched_get_route_name  # type: ignore[assignment]


def setup_metrics(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
