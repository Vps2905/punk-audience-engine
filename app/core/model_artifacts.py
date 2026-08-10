from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


DEFAULT_LOCAL_SEMANTIC_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
MODEL_ARTIFACT_FORMAT_VERSION = 1
MODEL_ARTIFACT_MANIFEST = "artifact-manifest.json"
_MANIFEST_TEMP_PREFIX = ".artifact-manifest-"


class ModelArtifactIntegrityError(RuntimeError):
    """Raised when a local model artifact is absent or untrusted."""


@dataclass(frozen=True)
class LocalSemanticModelArtifact:
    path: Path
    model_name: str
    revision: str
    embedding_dimension: int
    tree_sha256: str


def configured_local_semantic_model(
    environment: Mapping[str, str],
) -> tuple[str, str]:
    configured_name = environment.get("LOCAL_SEMANTIC_MODEL")
    configured_revision = environment.get("LOCAL_SEMANTIC_MODEL_REVISION")
    model_name = str(
        DEFAULT_LOCAL_SEMANTIC_MODEL
        if configured_name is None
        else configured_name
    ).strip()
    revision = str(
        DEFAULT_LOCAL_SEMANTIC_MODEL_REVISION
        if configured_revision is None
        else configured_revision
    ).strip().lower()

    if not model_name or model_name.startswith(("/", ".")):
        raise ValueError(
            "LOCAL_SEMANTIC_MODEL must be a non-local model identifier."
        )
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError(
            "LOCAL_SEMANTIC_MODEL_REVISION must be an immutable "
            "40-character commit SHA."
        )
    return model_name, revision


def local_semantic_model_artifact_path(
    cache_dir: str | Path,
    model_name: str,
    revision: str,
) -> Path:
    identity = hashlib.sha256(
        f"{model_name}@{revision}".encode("utf-8")
    ).hexdigest()[:24]
    return Path(cache_dir).expanduser() / "artifacts" / identity


def _artifact_file_records(artifact_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        artifact_path.rglob("*"),
        key=lambda candidate: candidate.as_posix(),
    ):
        relative = path.relative_to(artifact_path)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if relative.name == MODEL_ARTIFACT_MANIFEST:
            continue
        if relative.name.startswith(_MANIFEST_TEMP_PREFIX):
            continue
        if path.is_symlink():
            raise ModelArtifactIntegrityError(
                "Local semantic model artifact contains a symbolic link."
            )
        if not path.is_file():
            continue

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not records:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact contains no model files."
        )
    return records


def _artifact_tree_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_local_semantic_model_manifest(
    *,
    artifact_path: Path,
    model_name: str,
    revision: str,
    embedding_dimension: int,
) -> LocalSemanticModelArtifact:
    if embedding_dimension < 1:
        raise ModelArtifactIntegrityError(
            "Local semantic model embedding dimension is invalid."
        )
    artifact_path = artifact_path.resolve()
    records = _artifact_file_records(artifact_path)
    tree_sha256 = _artifact_tree_sha256(records)
    manifest = {
        "artifact_format_version": MODEL_ARTIFACT_FORMAT_VERSION,
        "model_name": model_name,
        "revision": revision,
        "embedding_dimension": embedding_dimension,
        "tree_sha256": tree_sha256,
        "files": records,
    }

    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=artifact_path,
            prefix=_MANIFEST_TEMP_PREFIX,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            artifact_path / MODEL_ARTIFACT_MANIFEST,
        )
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return LocalSemanticModelArtifact(
        path=artifact_path,
        model_name=model_name,
        revision=revision,
        embedding_dimension=embedding_dimension,
        tree_sha256=tree_sha256,
    )


def validate_local_semantic_model_artifact(
    *,
    cache_dir: str | Path,
    model_name: str,
    revision: str,
) -> LocalSemanticModelArtifact:
    artifact_path = local_semantic_model_artifact_path(
        cache_dir,
        model_name,
        revision,
    ).resolve()
    manifest_path = artifact_path / MODEL_ARTIFACT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact manifest is unavailable."
        ) from exc

    if not isinstance(manifest, dict):
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact manifest is invalid."
        )
    if manifest.get("artifact_format_version") != MODEL_ARTIFACT_FORMAT_VERSION:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact format is unsupported."
        )
    if (
        manifest.get("model_name") != model_name
        or manifest.get("revision") != revision
    ):
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact identity does not match configuration."
        )

    expected_records = manifest.get("files")
    actual_records = _artifact_file_records(artifact_path)
    if expected_records != actual_records:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact file integrity check failed."
        )
    actual_tree_sha256 = _artifact_tree_sha256(actual_records)
    if manifest.get("tree_sha256") != actual_tree_sha256:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact tree integrity check failed."
        )
    try:
        embedding_dimension = int(manifest["embedding_dimension"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact dimension is invalid."
        ) from exc
    if embedding_dimension < 1:
        raise ModelArtifactIntegrityError(
            "Local semantic model artifact dimension is invalid."
        )

    return LocalSemanticModelArtifact(
        path=artifact_path,
        model_name=model_name,
        revision=revision,
        embedding_dimension=embedding_dimension,
        tree_sha256=actual_tree_sha256,
    )
