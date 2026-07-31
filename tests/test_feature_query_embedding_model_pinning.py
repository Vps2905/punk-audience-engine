from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.audience_feature_proposal_service import (
    FeatureQueryEmbeddingService,
)


class FakeSentenceTransformer:
    init_calls = []
    encode_calls = []

    def __init__(self, model_name, **kwargs):
        self.init_calls.append((model_name, kwargs))

    def encode(self, texts, **kwargs):
        self.encode_calls.append((texts, kwargs))
        vector = np.zeros((len(texts), 384), dtype=float)
        vector[:, 0] = 1.0
        return vector


def _feature_set(**overrides):
    values = {
        "model_backend": "sentence_transformers",
        "model_name": "organization/multilingual-model",
        "model_version": "immutable-revision-123",
        "embedding_dimension": 384,
        "lineage": {
            "embedding_model_spec": {
                "query_prefix": "query: ",
                "normalize_embeddings": True,
            }
        },
    }
    values.update(overrides)
    return values


def test_query_uses_same_pinned_revision_prefix_and_cached_model(
    monkeypatch,
):
    FakeSentenceTransformer.init_calls.clear()
    FakeSentenceTransformer.encode_calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        ),
    )
    service = FeatureQueryEmbeddingService()

    first = service.encode("evening restaurants", _feature_set())
    second = service.encode("coffee visitors", _feature_set())

    assert first.shape == (384,)
    assert second.shape == (384,)
    assert len(FakeSentenceTransformer.init_calls) == 1
    model_name, kwargs = FakeSentenceTransformer.init_calls[0]
    assert model_name == "organization/multilingual-model"
    assert kwargs["revision"] == "immutable-revision-123"
    assert kwargs["trust_remote_code"] is False
    assert FakeSentenceTransformer.encode_calls[0][0] == [
        "query: evening restaurants"
    ]
    assert (
        FakeSentenceTransformer.encode_calls[0][1][
            "normalize_embeddings"
        ]
        is True
    )


def test_query_rejects_mutable_or_missing_model_revision(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=FakeSentenceTransformer
        ),
    )
    service = FeatureQueryEmbeddingService()

    with pytest.raises(ValueError, match="immutable"):
        service.encode(
            "audience",
            _feature_set(model_version="main"),
        )
