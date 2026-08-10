from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

from app.core.model_artifacts import (
    configured_local_semantic_model,
    local_semantic_model_artifact_path,
    validate_local_semantic_model_artifact,
    write_local_semantic_model_manifest,
)


def prefetch_local_semantic_model(
    *,
    environment: dict[str, str] | None = None,
    cache_dir: str | Path | None = None,
    model_factory: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    selected_environment = dict(
        os.environ if environment is None else environment
    )
    model_name, revision = configured_local_semantic_model(
        selected_environment
    )
    selected_cache = Path(
        cache_dir
        or selected_environment.get("LOCAL_SEMANTIC_MODEL_CACHE")
        or selected_environment.get("SENTENCE_TRANSFORMERS_HOME")
        or selected_environment.get("HF_HOME")
        or "/opt/models"
    ).expanduser()
    selected_cache.mkdir(parents=True, exist_ok=True)
    artifact_path = local_semantic_model_artifact_path(
        selected_cache,
        model_name,
        revision,
    )
    artifact_path.mkdir(parents=True, exist_ok=True)

    if snapshot_downloader is None:
        from huggingface_hub import snapshot_download

        snapshot_downloader = snapshot_download

    snapshot_downloader(
        repo_id=model_name,
        revision=revision,
        cache_dir=str(selected_cache),
        local_dir=str(artifact_path),
    )

    if model_factory is None:
        from sentence_transformers import SentenceTransformer

        model_factory = SentenceTransformer

    model = model_factory(
        str(artifact_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension < 1:
        raise RuntimeError("Local semantic model returned an invalid dimension.")

    write_local_semantic_model_manifest(
        artifact_path=artifact_path,
        model_name=model_name,
        revision=revision,
        embedding_dimension=dimension,
    )
    artifact = validate_local_semantic_model_artifact(
        cache_dir=selected_cache,
        model_name=model_name,
        revision=revision,
    )

    return {
        "status": "local_semantic_model_prefetched",
        "model_name": model_name,
        "model_revision": revision,
        "embedding_dimension": dimension,
        "artifact_path": str(artifact.path),
        "artifact_tree_sha256": artifact.tree_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch the immutable local semantic model artifact."
    )
    parser.add_argument("--cache-dir", default=None)
    arguments = parser.parse_args()
    result = prefetch_local_semantic_model(cache_dir=arguments.cache_dir)
    print(f"MODEL_NAME={result['model_name']}")
    print(f"MODEL_REVISION={result['model_revision']}")
    print(f"EMBEDDING_DIMENSION={result['embedding_dimension']}")
    print(f"MODEL_ARTIFACT_PATH={result['artifact_path']}")
    print(f"MODEL_ARTIFACT_SHA256={result['artifact_tree_sha256']}")
    print("MODEL_PREFETCH_OK")


if __name__ == "__main__":
    main()
