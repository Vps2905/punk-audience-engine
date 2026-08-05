#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.services.production_governed_constraint_taxonomy_service import (
    build_engineering_multilingual_taxonomy_artifacts,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"STOP: invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"STOP: expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an engineering-only governed multilingual taxonomy and "
            "immutable benchmark envelope from a checksum-pinned reviewed "
            "language pack. No registration, routing, or export occurs."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--language-pack", required=True, type=Path)
    parser.add_argument("--taxonomy-output", required=True, type=Path)
    parser.add_argument("--envelope-output", required=True, type=Path)
    parser.add_argument("--expected-language-pack-sha256", required=True)
    parser.add_argument(
        "--confirm-engineering-benchmark-only",
        action="store_true",
        help=(
            "Required acknowledgement that the generated taxonomy is not "
            "production-certified until native-human review is complete."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.confirm_engineering_benchmark_only:
        raise SystemExit(
            "STOP: --confirm-engineering-benchmark-only is required."
        )

    dataset_path = args.dataset.expanduser().resolve()
    pack_path = args.language_pack.expanduser().resolve()
    taxonomy_path = args.taxonomy_output.expanduser().resolve()
    envelope_path = args.envelope_output.expanduser().resolve()

    input_paths = {dataset_path, pack_path}
    output_paths = {taxonomy_path, envelope_path}
    if len(output_paths) != 2 or input_paths.intersection(output_paths):
        raise SystemExit(
            "STOP: inputs and outputs must be four distinct file paths."
        )
    if not dataset_path.is_file():
        raise SystemExit(f"STOP: immutable dataset not found: {dataset_path}")
    if not pack_path.is_file():
        raise SystemExit(f"STOP: reviewed language pack not found: {pack_path}")

    expected_pack_sha = args.expected_language_pack_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_pack_sha):
        raise SystemExit("STOP: expected language-pack SHA-256 is invalid.")
    actual_pack_sha = _sha256_file(pack_path)
    if actual_pack_sha != expected_pack_sha:
        raise SystemExit(
            "STOP: language-pack checksum mismatch: "
            f"expected={expected_pack_sha} actual={actual_pack_sha}"
        )

    original_dataset_file_sha = _sha256_file(dataset_path)
    dataset_payload = _load_json(dataset_path)
    pack_payload = _load_json(pack_path)
    artifacts = build_engineering_multilingual_taxonomy_artifacts(
        dataset_payload=dataset_payload,
        language_pack=pack_payload,
        language_pack_sha256=actual_pack_sha,
    )

    _write_json_atomic(taxonomy_path, artifacts.taxonomy_payload)
    _write_json_atomic(envelope_path, artifacts.dataset_envelope)

    current_dataset_file_sha = _sha256_file(dataset_path)
    if current_dataset_file_sha != original_dataset_file_sha:
        raise SystemExit("STOP: immutable dataset file changed")

    result = {
        "status": "engineering_exact_alias_taxonomy_created",
        "taxonomy_id": artifacts.taxonomy.taxonomy_id,
        "taxonomy_version": artifacts.taxonomy.version,
        "taxonomy_fingerprint": artifacts.taxonomy.fingerprint,
        "location_count": artifacts.location_count,
        "category_count": artifacts.category_count,
        "daypart_count": artifacts.daypart_count,
        "canonical_dataset_fingerprint": (
            artifacts.canonical_dataset_fingerprint
        ),
        "immutable_dataset_file_sha256": current_dataset_file_sha,
        "language_pack_sha256": actual_pack_sha,
        "taxonomy_output": str(taxonomy_path),
        "taxonomy_output_sha256": _sha256_file(taxonomy_path),
        "envelope_output": str(envelope_path),
        "envelope_output_sha256": _sha256_file(envelope_path),
        "oracle_case_constraints_used": False,
        "native_human_production_signoff_pending": True,
        "registration_allowed": False,
        "production_routing_enabled": False,
        "activation_or_export_performed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
