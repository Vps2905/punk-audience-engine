from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from app.models.production_agent_security_contracts import (
    AgentAuthorizationContext,
    AgentSecurityCertificationRequest,
    AgentSecurityPrincipal,
)
from app.models.production_audience_retrieval_contracts import (
    GovernedAudienceRetrievalRequest,
    ProductionRetrievalModelBinding,
)
from app.models.production_bounded_autonomy_contracts import (
    AutonomyBudget,
    AutonomyGoal,
)
from app.models.production_bounded_autonomy_functional_shadow_contracts import (
    FunctionalShadowPolicy,
    FunctionalShadowRunRequest,
)
from app.models.production_feature_build_contracts import EmbeddingModelSpec
from app.models.production_security_contracts import (
    ProductionSecurityAssessmentRequest,
)
from app.services.pgvector_audience_feature_store_service import (
    PgvectorAudienceFeatureStore,
)
from app.services.production_agent_security_certification_service import (
    ProductionAgentSecurityCertificationService,
)
from app.services.production_bounded_autonomy_certification_service import (
    ProductionBoundedAutonomyCertificationService,
)
from app.services.production_bounded_autonomy_functional_shadow_service import (
    ProductionBoundedAutonomyFunctionalShadowService,
    ProductionFeatureSnapshotReaderAdapter,
)
from app.services.production_dual_model_candidate_retrieval_service import (
    ProductionDualModelCandidateRetrievalService,
)
from app.services.production_embedding_model_registry_service import (
    ProductionEmbeddingModelRegistryService,
)
from app.services.production_feature_snapshot_reader_service import (
    ProductionFeatureSnapshotReaderService,
)
from app.services.production_governed_audience_retrieval_service import (
    ProductionGovernedAudienceRetrievalService,
)
from app.services.production_governed_constraint_taxonomy_service import (
    load_governed_constraint_taxonomy,
)
from app.services.production_multilingual_constraint_canonicalization_service import (
    ProductionMultilingualConstraintCanonicalizationService,
)
from app.services.production_security_hardening_service import (
    ProductionSecurityPostureService,
)


_READY = "engineering_preview_ready"


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object.")
    return json.loads(json.dumps(dict(value)))


def _tuple(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a JSON array.")
    return tuple(str(item) for item in value)


def _model(value: Any, *, label: str) -> EmbeddingModelSpec:
    payload = _mapping(value, label=label)
    payload.pop("model_fingerprint", None)
    return EmbeddingModelSpec(**payload)


def _binding(value: Any, *, label: str) -> ProductionRetrievalModelBinding:
    payload = _mapping(value, label=label)
    payload["model"] = _model(payload.get("model"), label=f"{label}.model")
    return ProductionRetrievalModelBinding(**payload)


class ProductionModule5FunctionalEvidenceOrchestrator:
    """Execute Module 5.9 with validated lineage and real read-only adapters."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        source_certification_validator: Any | None = None,
        functional_service_factory: Any | None = None,
    ) -> None:
        self.environment = dict(os.environ if environment is None else environment)
        self._source_certification_validator = (
            source_certification_validator
            or ProductionBoundedAutonomyCertificationService().validate_report
        )
        self._functional_service_factory = functional_service_factory

    def run(
        self,
        *,
        source_certification_report: Mapping[str, Any],
        functional_input: Mapping[str, Any],
        taxonomy_payload: Mapping[str, Any],
        legacy_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        source = self._source_certification_validator(
            _mapping(
                source_certification_report,
                label="source_certification_report",
            )
        )
        self._require_ready_source_certification(source)
        configuration = _mapping(functional_input, label="functional_input")
        goal = self._goal(configuration.get("goal"))
        if source.get("tenant_id") != goal.tenant_id:
            raise ValueError("Module 5.8 and Module 5.9 tenant lineage mismatch.")

        retrieval = self._retrieval_request(
            value=configuration.get("retrieval"),
            taxonomy_payload=taxonomy_payload,
            goal=goal,
        )
        request = self._functional_request(
            value=configuration.get("run"),
            goal=goal,
            retrieval=retrieval,
            source_fingerprint=source[
                "bounded_autonomy_certification_fingerprint"
            ],
        )
        authorization = self._authorization_context(
            configuration.get("authorization_context"),
            goal=goal,
        )
        baseline = configuration.get("baseline_snapshot")
        if baseline is not None:
            baseline = _mapping(baseline, label="baseline_snapshot")
        policy_value = configuration.get("policy")
        policy = (
            FunctionalShadowPolicy(
                **_mapping(policy_value, label="functional_shadow_policy")
            )
            if policy_value is not None
            else None
        )
        service = self._service(policy=policy)
        report = service.run(
            request=request,
            goal=goal,
            retrieval_request=retrieval,
            legacy_result=_mapping(legacy_result, label="legacy_result"),
            authorization_context=authorization,
            baseline_snapshot=baseline,
        )
        validated = service.validate_report(report)
        if (
            validated["request"]["source_certification_report_fingerprint"]
            != source["bounded_autonomy_certification_fingerprint"]
        ):
            raise ValueError("Module 5.9 source-certification lineage mismatch.")
        return validated

    def _service(
        self,
        *,
        policy: FunctionalShadowPolicy | None,
    ) -> ProductionBoundedAutonomyFunctionalShadowService:
        if self._functional_service_factory is not None:
            value = self._functional_service_factory(
                environment=self.environment,
                policy=policy,
            )
            if not isinstance(
                value,
                ProductionBoundedAutonomyFunctionalShadowService,
            ):
                raise TypeError(
                    "Functional service factory returned an invalid service."
                )
            return value

        database_url = str(
            self.environment.get("AUDIENCE_FEATURE_DATABASE_URL") or ""
        ).strip()
        if not database_url:
            raise RuntimeError(
                "AUDIENCE_FEATURE_DATABASE_URL is required for real Module 5.9 "
                "evidence generation."
            )
        registry = ProductionEmbeddingModelRegistryService(database_url)
        retrieval = ProductionGovernedAudienceRetrievalService(
            canonicalizer=(
                ProductionMultilingualConstraintCanonicalizationService(
                    model_registry=registry,
                )
            ),
            candidate_retrieval=ProductionDualModelCandidateRetrievalService(
                feature_store=PgvectorAudienceFeatureStore(database_url),
                model_registry=registry,
            ),
        )
        source_reader = ProductionFeatureSnapshotReaderAdapter(
            ProductionFeatureSnapshotReaderService(database_url=database_url)
        )
        return ProductionBoundedAutonomyFunctionalShadowService(
            source_reader=source_reader,
            retrieval_service=retrieval,
            policy=policy,
            environment=self.environment,
        )

    def _goal(self, value: Any) -> AutonomyGoal:
        payload = _mapping(value, label="goal")
        payload["requested_outcomes"] = _tuple(
            payload.get("requested_outcomes"),
            label="goal.requested_outcomes",
        )
        payload["constraints"] = _tuple(
            payload.get("constraints"),
            label="goal.constraints",
        )
        budget = payload.get("budget")
        if budget is not None:
            payload["budget"] = AutonomyBudget(
                **_mapping(budget, label="goal.budget")
            )
        return AutonomyGoal(**payload)

    def _retrieval_request(
        self,
        *,
        value: Any,
        taxonomy_payload: Mapping[str, Any],
        goal: AutonomyGoal,
    ) -> GovernedAudienceRetrievalRequest:
        payload = _mapping(value, label="retrieval")
        loaded = load_governed_constraint_taxonomy(
            _mapping(taxonomy_payload, label="taxonomy_payload"),
            allow_engineering_scope=False,
            require_native_human_review=True,
        )
        payload["tenant_id"] = goal.tenant_id
        payload["query_text"] = goal.objective
        payload["execution_mode"] = "historical_preview"
        payload["taxonomy"] = loaded.taxonomy
        payload["canonicalizer_model"] = _model(
            payload.get("canonicalizer_model"),
            label="retrieval.canonicalizer_model",
        )
        payload["primary_model"] = _binding(
            payload.get("primary_model"),
            label="retrieval.primary_model",
        )
        payload["complementary_model"] = _binding(
            payload.get("complementary_model"),
            label="retrieval.complementary_model",
        )
        for field in (
            "requested_locations",
            "requested_categories",
            "requested_dayparts",
            "exclusions",
        ):
            payload[field] = _tuple(
                payload.get(field),
                label=f"retrieval.{field}",
            )
        return GovernedAudienceRetrievalRequest(**payload)

    def _functional_request(
        self,
        *,
        value: Any,
        goal: AutonomyGoal,
        retrieval: GovernedAudienceRetrievalRequest,
        source_fingerprint: str,
    ) -> FunctionalShadowRunRequest:
        payload = _mapping(value, label="run")
        payload.update(
            tenant_id=goal.tenant_id,
            request_id=goal.request_id,
            feature_set_id=retrieval.primary_model.feature_set_id,
            feature_set_version=retrieval.primary_model.feature_set_version,
            source_certification_report_fingerprint=source_fingerprint,
        )
        return FunctionalShadowRunRequest(**payload)

    def _authorization_context(
        self,
        value: Any,
        *,
        goal: AutonomyGoal,
    ) -> AgentAuthorizationContext:
        payload = _mapping(value, label="authorization_context")
        principal_payload = _mapping(
            payload.get("principal"),
            label="authorization_context.principal",
        )
        principal_payload["scopes"] = _tuple(
            principal_payload.get("scopes"),
            label="authorization_context.principal.scopes",
        )
        principal_payload["allowed_capability_ids"] = _tuple(
            principal_payload.get("allowed_capability_ids"),
            label="authorization_context.principal.allowed_capability_ids",
        )
        principal = AgentSecurityPrincipal(**principal_payload)
        payload["principal"] = principal
        if payload.get("request_id") != goal.request_id:
            raise ValueError("Authorization request lineage does not match goal.")
        return AgentAuthorizationContext(**payload)

    @staticmethod
    def _require_ready_source_certification(source: Mapping[str, Any]) -> None:
        review = dict(source.get("review") or {})
        if (
            source.get("status") != _READY
            or review.get("eligible_for_staging_review") is not True
            or review.get("live_cutover_authorized") is not False
        ):
            raise ValueError(
                "Module 5.8 certification is not eligible for staging review."
            )


class ProductionModule5AgentSecurityEvidenceOrchestrator:
    """Assess the actual environment and certify Module 5.10 only if it passes."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        functional_validator: Any | None = None,
        posture_service: Any | None = None,
        certification_service: Any | None = None,
    ) -> None:
        self.environment = dict(os.environ if environment is None else environment)
        self._functional_validator = functional_validator
        self._posture_service = posture_service or ProductionSecurityPostureService()
        self._certification_service = (
            certification_service or ProductionAgentSecurityCertificationService()
        )

    def run(
        self,
        *,
        tenant_id: str,
        assessment_id: str,
        certification_id: str,
        functional_shadow_report: Mapping[str, Any],
        evaluation_epoch_seconds: int | None = None,
    ) -> dict[str, Any]:
        validator = self._functional_validator or (
            ProductionBoundedAutonomyFunctionalShadowService(
                environment={}
            ).validate_report
        )
        functional = validator(
            _mapping(functional_shadow_report, label="functional_shadow_report")
        )
        if functional.get("status") != _READY:
            raise ValueError("Module 5.9 functional evidence is not ready.")
        if functional.get("request", {}).get("tenant_id") != tenant_id:
            raise ValueError("Module 5.9 and Module 5.10 tenant lineage mismatch.")

        posture = self._posture_service.assess(
            request=ProductionSecurityAssessmentRequest(
                tenant_id=tenant_id,
                assessment_id=assessment_id,
                execution_mode="production",
            ),
            environment=self.environment,
        )
        if posture.get("posture_status") != "pass":
            return {
                "status": "engineering_preview_blocked",
                "reason_code": "production_security_posture_failed",
                "failed_control_codes": list(
                    posture.get("failed_control_codes") or []
                ),
                "security_posture_report": posture,
                "agent_security_certification_report": None,
                "production_ready": False,
                "live_cutover_authorized": False,
            }

        evaluated = int(evaluation_epoch_seconds or time.time())
        request = AgentSecurityCertificationRequest(
            tenant_id=tenant_id,
            certification_id=certification_id,
            evaluation_epoch_seconds=evaluated,
            source_functional_shadow_report_fingerprint=functional[
                "functional_shadow_report_fingerprint"
            ],
            source_security_posture_fingerprint=posture[
                "security_posture_fingerprint"
            ],
        )
        certification = self._certification_service.certify(
            request=request,
            functional_shadow_report=functional,
            security_posture_report=posture,
        )
        return {
            "status": certification["status"],
            "reason_code": (
                "agent_security_certification_ready"
                if certification["status"] == _READY
                else "agent_security_certification_blocked"
            ),
            "failed_control_codes": [],
            "security_posture_report": posture,
            "agent_security_certification_report": certification,
            "production_ready": False,
            "live_cutover_authorized": False,
        }
