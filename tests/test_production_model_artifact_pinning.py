from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest

from app.core.model_artifacts import (
    DEFAULT_LOCAL_SEMANTIC_MODEL,
    DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
    MODEL_ARTIFACT_MANIFEST,
    ModelArtifactIntegrityError,
    configured_local_semantic_model,
    local_semantic_model_artifact_path,
    validate_local_semantic_model_artifact,
    write_local_semantic_model_manifest,
)
from scripts.prefetch_local_semantic_model import (
    prefetch_local_semantic_model,
)


def test_default_local_semantic_model_is_immutable():
    model_name, revision = configured_local_semantic_model({})
    assert model_name == DEFAULT_LOCAL_SEMANTIC_MODEL
    assert revision == DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


@pytest.mark.parametrize("revision", ["", "main", "latest", "v1"])
def test_mutable_local_semantic_revision_is_rejected(revision):
    with pytest.raises(ValueError, match="immutable"):
        configured_local_semantic_model(
            {
                "LOCAL_SEMANTIC_MODEL": DEFAULT_LOCAL_SEMANTIC_MODEL,
                "LOCAL_SEMANTIC_MODEL_REVISION": revision,
            }
        )


def test_prefetch_uses_exact_revision_and_disables_remote_code(tmp_path):
    download_calls = []
    model_calls = []

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 384

    def factory(model_name, **kwargs):
        model_calls.append((model_name, kwargs))
        return FakeModel()

    def downloader(**kwargs):
        download_calls.append(kwargs)
        artifact_root = tmp_path / "artifacts"
        candidates = list(artifact_root.iterdir())
        assert len(candidates) == 1
        (candidates[0] / "config.json").write_text(
            '{"model_type": "bert"}\n',
            encoding="utf-8",
        )

    report = prefetch_local_semantic_model(
        environment={},
        cache_dir=tmp_path,
        model_factory=factory,
        snapshot_downloader=downloader,
    )

    assert report["model_revision"] == DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION
    assert report["embedding_dimension"] == 384
    artifact_path = local_semantic_model_artifact_path(
        tmp_path,
        DEFAULT_LOCAL_SEMANTIC_MODEL,
        DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
    ).resolve()
    assert download_calls == [
        {
            "repo_id": DEFAULT_LOCAL_SEMANTIC_MODEL,
            "revision": DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
            "cache_dir": str(tmp_path),
            "local_dir": str(artifact_path),
        }
    ]
    assert model_calls == [
        (
            str(artifact_path),
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert (artifact_path / MODEL_ARTIFACT_MANIFEST).is_file()
    assert report["artifact_tree_sha256"] == (
        validate_local_semantic_model_artifact(
            cache_dir=tmp_path,
            model_name=DEFAULT_LOCAL_SEMANTIC_MODEL,
            revision=DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
        ).tree_sha256
    )


def test_artifact_manifest_rejects_tampered_model_file(tmp_path):
    artifact_path = local_semantic_model_artifact_path(
        tmp_path,
        DEFAULT_LOCAL_SEMANTIC_MODEL,
        DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
    )
    artifact_path.mkdir(parents=True)
    config_path = artifact_path / "config.json"
    config_path.write_text("reviewed", encoding="utf-8")
    write_local_semantic_model_manifest(
        artifact_path=artifact_path,
        model_name=DEFAULT_LOCAL_SEMANTIC_MODEL,
        revision=DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
        embedding_dimension=384,
    )

    config_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ModelArtifactIntegrityError, match="integrity"):
        validate_local_semantic_model_artifact(
            cache_dir=tmp_path,
            model_name=DEFAULT_LOCAL_SEMANTIC_MODEL,
            revision=DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
        )


def test_runtime_loader_requires_cached_exact_revision(monkeypatch):
    from app.services.local_semantic_intent_service import (
        LocalSemanticIntentService,
    )

    calls = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL_REVISION", raising=False)
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL_CACHE", raising=False)
    monkeypatch.setattr(LocalSemanticIntentService, "_model", None)
    monkeypatch.setattr(LocalSemanticIntentService, "_model_identity", None)

    LocalSemanticIntentService._get_model()

    assert calls == [
        (
            DEFAULT_LOCAL_SEMANTIC_MODEL,
            {
                "revision": DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


def test_runtime_loader_uses_verified_local_artifact(monkeypatch, tmp_path):
    from app.services.local_semantic_intent_service import (
        LocalSemanticIntentService,
    )

    calls = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    artifact_path = local_semantic_model_artifact_path(
        tmp_path,
        DEFAULT_LOCAL_SEMANTIC_MODEL,
        DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
    )
    artifact_path.mkdir(parents=True)
    (artifact_path / "config.json").write_text(
        '{"model_type": "bert"}\n',
        encoding="utf-8",
    )
    write_local_semantic_model_manifest(
        artifact_path=artifact_path,
        model_name=DEFAULT_LOCAL_SEMANTIC_MODEL,
        revision=DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION,
        embedding_dimension=384,
    )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL_REVISION", raising=False)
    monkeypatch.setenv("LOCAL_SEMANTIC_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(LocalSemanticIntentService, "_model", None)
    monkeypatch.setattr(LocalSemanticIntentService, "_model_identity", None)

    LocalSemanticIntentService._get_model()

    assert calls == [
        (
            str(artifact_path.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


def test_runtime_loader_fails_closed_without_artifact_manifest(
    monkeypatch,
    tmp_path,
):
    from app.services.local_semantic_intent_service import (
        LocalSemanticIntentService,
        LocalSemanticIntentUnavailableError,
    )

    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_SEMANTIC_MODEL_REVISION", raising=False)
    monkeypatch.setenv("LOCAL_SEMANTIC_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(LocalSemanticIntentService, "_model", None)
    monkeypatch.setattr(LocalSemanticIntentService, "_model_identity", None)

    with pytest.raises(
        LocalSemanticIntentUnavailableError,
        match="ModelArtifactIntegrityError",
    ):
        LocalSemanticIntentService._get_model()
