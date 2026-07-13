from app.api.embedding_routes import router
from app.core.api_key_auth import (
    require_audience_api_key,
)


def test_embedding_router_requires_api_key():
    dependencies = [
        getattr(item, "dependency", None)
        for item in router.dependencies
    ]

    assert require_audience_api_key in dependencies
