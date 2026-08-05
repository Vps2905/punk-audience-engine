from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models.audience_feature_contracts import stable_digest
from app.models.production_module2_completion_contracts import (
    Module2IndexManifest,
    Module2IndexValidationEvidence,
)
from app.services.production_module2_index_lifecycle_service import (
    InMemoryModule2IndexLifecycleRepository,
    ProductionModule2IndexLifecycleService,
)

NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)


def _manifest(version=1, mode="offline_evaluation"):
    return Module2IndexManifest(
        tenant_id="tenant-a",
        index_id="module2-global-index",
        index_version=version,
        taxonomy_fingerprint="a" * 64,
        primary_model_fingerprint="b" * 64,
        complementary_model_fingerprint="c" * 64,
        canonicalizer_model_fingerprint="d" * 64,
        feature_set_id="feature-set-a",
        feature_set_version=version,
        source_fingerprint="e" * 64,
        data_use_mode=mode,
        embedding_dimension=384,
        expected_document_count=100,
        built_by="index_builder",
        created_at=NOW,
    )


def _validation(manifest):
    return Module2IndexValidationEvidence(
        manifest_fingerprint=manifest.fingerprint,
        validated_by="index_validator",
        validated_at=NOW,
        observed_document_count=100,
        embedding_dimension=384,
        missing_vector_count=0,
        invalid_vector_count=0,
        duplicate_document_count=0,
        tenant_isolation_passed=True,
        model_binding_passed=True,
        taxonomy_binding_passed=True,
        source_binding_passed=True,
        recall_at_10=0.99,
        p95_latency_ms=40.0,
        checksum_sha256="f" * 64,
    )


def _cert(evidence=True, production=False):
    report = {
        "module2_evidence_ready": evidence,
        "production_certification_ready": production,
        "model_registration_performed": False,
        "production_routing_enabled": False,
        "automatic_proposal_creation_enabled": False,
        "activation_or_export_performed": False,
        "downstream_export_enabled": False,
    }
    report["report_fingerprint"] = stable_digest(report)
    return report


def _shadow(ready=True):
    report = {
        "shadow_release_ready": ready,
        "raw_query_stored": False,
        "raw_identifiers_stored": False,
        "candidate_output_routed_to_user": False,
        "production_routing_enabled": False,
        "automatic_proposal_creation_enabled": False,
        "activation_or_export_performed": False,
    }
    report["report_fingerprint"] = stable_digest(report)
    return report


def test_index_lifecycle_reaches_shadow_without_enabling_production_routing():
    repository = InMemoryModule2IndexLifecycleRepository()
    service = ProductionModule2IndexLifecycleService(repository=repository)
    manifest = _manifest()

    assert service.register_candidate(manifest)["status"] == "candidate"
    assert service.mark_building(manifest=manifest, worker_id="worker")["status"] == "building"
    assert service.mark_built(
        manifest=manifest,
        observed_document_count=100,
        artifact_checksum="9" * 64,
    )["status"] == "built"
    assert service.record_validation(
        manifest=manifest,
        validation=_validation(manifest),
    )["status"] == "validated"
    shadow = service.promote_to_shadow(
        manifest=manifest,
        approved_by="release_owner",
        certification_report=_cert(),
    )
    assert shadow["status"] == "shadow"
    assert shadow["latest_evidence"]["production_routing_enabled"] is False


def test_active_promotion_requires_production_data_certification_shadow_and_explicit_flag():
    repository = InMemoryModule2IndexLifecycleRepository()
    service = ProductionModule2IndexLifecycleService(repository=repository)
    manifest = _manifest(mode="production")
    service.register_candidate(manifest)
    service.mark_building(manifest=manifest, worker_id="worker")
    service.mark_built(
        manifest=manifest,
        observed_document_count=100,
        artifact_checksum="9" * 64,
    )
    service.record_validation(manifest=manifest, validation=_validation(manifest))
    service.promote_to_shadow(
        manifest=manifest,
        approved_by="release_owner",
        certification_report=_cert(),
    )

    with pytest.raises(RuntimeError, match="Explicit production routing"):
        service.promote_to_active(
            manifest=manifest,
            approved_by="operator",
            certification_report=_cert(production=True),
            shadow_report=_shadow(),
        )

    active = service.promote_to_active(
        manifest=manifest,
        approved_by="operator",
        certification_report=_cert(production=True),
        shadow_report=_shadow(),
        production_routing_enabled=True,
    )
    assert active["status"] == "active"
    assert active["latest_evidence"]["downstream_export_enabled"] is False


def test_incremental_updates_create_new_version_plan_not_in_place_mutation():
    service = ProductionModule2IndexLifecycleService(
        repository=InMemoryModule2IndexLifecycleRepository()
    )
    plan = service.plan_incremental_update(
        base_manifest=_manifest(),
        delta_source_fingerprint="8" * 64,
        insert_count=10,
        update_count=2,
        delete_count=1,
    )
    assert plan["requires_new_immutable_index_version"] is True
    assert plan["in_place_mutation_allowed"] is False
    assert plan["production_routing_enabled"] is False
