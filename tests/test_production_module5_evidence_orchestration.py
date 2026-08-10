from __future__ import annotations

import json
import stat
import time

import pytest

from app.core.atomic_evidence_writer import write_new_private_json
from app.models.production_audience_retrieval_contracts import (
    ConstraintTaxonomyEntry,
    GovernedConstraintTaxonomy,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
)
from app.services.production_module5_evidence_orchestration_service import (
    ProductionModule5AgentSecurityEvidenceOrchestrator,
    ProductionModule5FunctionalEvidenceOrchestrator,
)


TENANT = "tenant_a"
SOURCE_FINGERPRINT = "a" * 64
FUNCTIONAL_FINGERPRINT = "b" * 64
POSTURE_FINGERPRINT = "c" * 64

E5 = EmbeddingModelSpec(
    backend="sentence_transformers",
    model_name="intfloat/multilingual-e5-small",
    model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    normalize_embeddings=True,
    document_prefix="passage: ",
    query_prefix="query: ",
)
MINILM = EmbeddingModelSpec(
    backend="sentence_transformers",
    model_name=(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
    model_revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    dimension=384,
    normalize_embeddings=True,
)


class CapturingFunctionalService(
    ProductionBoundedAutonomyFunctionalShadowService
):
    def __init__(self):
        super().__init__(environment={})
        self.received = None

    def run(self, **kwargs):
        self.received = kwargs
        return {
            "status": "engineering_preview_ready",
            "request": kwargs["request"].to_record(),
            "review": {
                "eligible_for_staging_review": True,
                "live_cutover_authorized": False,
            },
            "functional_shadow_report_fingerprint": FUNCTIONAL_FINGERPRINT,
        }

    def validate_report(self, report):
        return dict(report)


def taxonomy_payload():
    taxonomy = GovernedConstraintTaxonomy(
        taxonomy_id="production-audience-taxonomy",
        version="reviewed-v1",
        reviewed_by="taxonomy-owner",
        locations=(
            ConstraintTaxonomyEntry("region_one", aliases=("Region One",)),
        ),
        categories=(
            ConstraintTaxonomyEntry("venue_one", aliases=("Venue One",)),
        ),
        dayparts=(
            ConstraintTaxonomyEntry("evening", aliases=("Evening",)),
        ),
    )
    value = taxonomy.to_safe_dict()
    value["lineage"] = {
        "approval_scope": "production_retrieval",
        "native_human_review_completed": True,
        "production_certification_status": "approved_for_production_retrieval",
        "oracle_case_constraints_used": False,
    }
    return value


def functional_input():
    now = int(time.time())
    return {
        "goal": {
            "tenant_id": TENANT,
            "request_id": "request-1",
            "goal_id": "goal-1",
            "objective": "Find Venue One audiences in Region One in the Evening.",
            "requested_outcomes": ["delivery_review"],
            "execution_mode": "shadow",
            "constraints": [],
        },
        "run": {
            "functional_run_id": "functional-run-1",
            "purpose": "audience_intelligence",
            "execution_mode": "shadow",
        },
        "retrieval": {
            "language": "en",
            "canonicalizer_model": E5.to_safe_dict(),
            "primary_model": {
                "role": "primary",
                "feature_set_id": "feature-primary",
                "feature_set_version": 1,
                "model": E5.to_safe_dict(),
                "initial_depth": 10,
                "expansion_depth": 30,
            },
            "complementary_model": {
                "role": "complementary",
                "feature_set_id": "feature-complementary",
                "feature_set_version": 1,
                "model": MINILM.to_safe_dict(),
                "initial_depth": 10,
                "expansion_depth": 10,
            },
            "requested_locations": ["Region One"],
            "requested_categories": ["Venue One"],
            "requested_dayparts": ["Evening"],
            "exclusions": [],
            "result_limit": 10,
        },
        "authorization_context": {
            "authorization_id": "authorization-1",
            "request_id": "request-1",
            "source_authentication_fingerprint": "d" * 64,
            "principal": {
                "tenant_id": TENANT,
                "principal_id": "bounded-agent-1",
                "principal_type": "bounded_agent",
                "authentication_method": "workload_identity",
                "scopes": ["audience:read", "audience:propose"],
                "allowed_capability_ids": [
                    "module1_provider_source_discovery",
                    "module1_public_source_discovery",
                    "module1_privacy_safe_aggregation",
                    "module2_semantic_retrieval",
                    "module3_cohort_strategy",
                    "module4_evolution_review",
                    "module5_governed_recommendation",
                    "module5_delivery_review",
                ],
                "issued_at_epoch_seconds": now - 10,
                "expires_at_epoch_seconds": now + 600,
            },
        },
    }


def source_report(*, ready=True, tenant=TENANT):
    return {
        "status": (
            "engineering_preview_ready"
            if ready
            else "engineering_preview_blocked"
        ),
        "tenant_id": tenant,
        "review": {
            "eligible_for_staging_review": ready,
            "live_cutover_authorized": False,
        },
        "bounded_autonomy_certification_fingerprint": SOURCE_FINGERPRINT,
    }


def test_functional_orchestrator_validates_lineage_and_builds_real_contracts():
    service = CapturingFunctionalService()
    orchestrator = ProductionModule5FunctionalEvidenceOrchestrator(
        environment={},
        source_certification_validator=lambda value: value,
        functional_service_factory=lambda **_kwargs: service,
    )

    report = orchestrator.run(
        source_certification_report=source_report(),
        functional_input=functional_input(),
        taxonomy_payload=taxonomy_payload(),
        legacy_result={"status": "completed"},
    )

    assert report["status"] == "engineering_preview_ready"
    assert service.received["goal"].tenant_id == TENANT
    assert service.received["retrieval_request"].execution_mode == (
        "historical_preview"
    )
    assert service.received["request"].source_certification_report_fingerprint == (
        SOURCE_FINGERPRINT
    )
    assert service.received["authorization_context"].principal.principal_type == (
        "bounded_agent"
    )


@pytest.mark.parametrize(
    "source",
    [source_report(ready=False), source_report(tenant="tenant_other")],
)
def test_functional_orchestrator_fails_closed_on_prerequisite_lineage(source):
    orchestrator = ProductionModule5FunctionalEvidenceOrchestrator(
        environment={},
        source_certification_validator=lambda value: value,
        functional_service_factory=lambda **_kwargs: CapturingFunctionalService(),
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            source_certification_report=source,
            functional_input=functional_input(),
            taxonomy_payload=taxonomy_payload(),
            legacy_result={"status": "completed"},
        )


class PostureService:
    def __init__(self, *, passed):
        self.passed = passed

    def assess(self, **_kwargs):
        return {
            "status": "engineering_preview_ready",
            "posture_status": "pass" if self.passed else "fail_closed",
            "failed_control_codes": [] if self.passed else ["database_tls"],
            "security_posture_fingerprint": POSTURE_FINGERPRINT,
        }


class SecurityCertificationService:
    def certify(self, *, request, **_kwargs):
        return {
            "status": "engineering_preview_ready",
            "agent_security_certification_fingerprint": "e" * 64,
            "request": request.to_record(),
        }


def functional_report():
    return {
        "status": "engineering_preview_ready",
        "request": {"tenant_id": TENANT},
        "functional_shadow_report_fingerprint": FUNCTIONAL_FINGERPRINT,
    }


def test_agent_security_orchestrator_writes_no_certification_on_failed_posture():
    result = ProductionModule5AgentSecurityEvidenceOrchestrator(
        environment={},
        functional_validator=lambda value: value,
        posture_service=PostureService(passed=False),
        certification_service=SecurityCertificationService(),
    ).run(
        tenant_id=TENANT,
        assessment_id="assessment-1",
        certification_id="certification-1",
        functional_shadow_report=functional_report(),
        evaluation_epoch_seconds=1,
    )

    assert result["status"] == "engineering_preview_blocked"
    assert result["agent_security_certification_report"] is None
    assert result["failed_control_codes"] == ["database_tls"]
    assert result["production_ready"] is False


def test_agent_security_orchestrator_certifies_only_after_posture_passes():
    result = ProductionModule5AgentSecurityEvidenceOrchestrator(
        environment={},
        functional_validator=lambda value: value,
        posture_service=PostureService(passed=True),
        certification_service=SecurityCertificationService(),
    ).run(
        tenant_id=TENANT,
        assessment_id="assessment-1",
        certification_id="certification-1",
        functional_shadow_report=functional_report(),
        evaluation_epoch_seconds=1,
    )

    assert result["status"] == "engineering_preview_ready"
    assert result["agent_security_certification_report"] is not None
    assert result["live_cutover_authorized"] is False


def test_private_evidence_writer_is_atomic_and_never_overwrites(tmp_path):
    output = tmp_path / "evidence.json"
    write_new_private_json(output, {"status": "ready"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "ready"
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_new_private_json(output, {"status": "replaced"})
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ready"
