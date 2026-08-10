from pathlib import Path
import re


CI_WORKFLOW = Path(".github/workflows/ci.yml")
DEPENDABOT = Path(".github/dependabot.yml")
CI_REQUIREMENTS = Path("requirements-ci.txt")
RUNBOOK = Path("docs/production_ci_supply_chain_runbook.md")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ci_runs_full_suite_with_disposable_postgres():
    workflow = _read(CI_WORKFLOW)
    for required in (
        "postgres:16-alpine",
        "POSTGRES_DB: audience_ci",
        "AUDIENCE_TEST_DATABASE_URL",
        "uv run --frozen python -m compileall -q app scripts tests",
        "uv run --frozen python -m pytest -q --junitxml=artifacts/pytest.xml",
    ):
        assert required in workflow

    assert "production" not in re.search(
        r"AUDIENCE_TEST_DATABASE_URL:\s*>-\s*([^\n]+)", workflow
    ).group(1).lower()


def test_ci_has_source_dependency_and_iac_security_gates():
    workflow = _read(CI_WORKFLOW)
    for required in (
        "uv sync --frozen --no-dev",
        'pip-audit --path "$RUNTIME_SITE_PACKAGES"',
        "bandit -r app scripts",
        "--severity-level high",
        "--confidence-level medium",
        "cfn-lint infrastructure/cloudformation/*.yaml",
    ):
        assert required in workflow


def test_ci_builds_smokes_scans_and_inventorys_container_without_push():
    workflow = _read(CI_WORKFLOW)
    for required in (
        "docker build --pull",
        "ALLOW_DEMO_ROUTES=false",
        "http://127.0.0.1:18000/health",
        "aquasecurity/trivy-action@",
        "severity: CRITICAL",
        "format: cyclonedx",
        "punk-audience-engine.sbom.cdx.json",
        "cosign sign-blob --yes",
        "punk-audience-engine.sbom.sigstore.json",
    ):
        assert required in workflow

    forbidden = (
        "docker push",
        "aws cloudformation deploy",
        "INFRASTRUCTURE_PRODUCTION_TRAFFIC_ENABLED=true",
        "MODULE5_PRODUCTION_ROUTING_ENABLED=true",
    )
    assert not any(value in workflow for value in forbidden)


def test_ci_uses_least_privilege_and_bounded_execution():
    workflow = _read(CI_WORKFLOW)
    assert "permissions:\n  contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    assert workflow.count("timeout-minutes:") == 3
    assert "workflow_dispatch:" in workflow


def test_all_third_party_actions_are_pinned_by_commit_sha():
    workflow = _read(CI_WORKFLOW)
    action_references = re.findall(r"uses:\s*[^\s@]+@([^\s#]+)", workflow)
    assert action_references
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in action_references)


def test_ci_tooling_dependencies_are_exactly_pinned():
    dependencies = []
    for raw_line in _read(CI_REQUIREMENTS).splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            dependencies.append(line)

    assert dependencies
    assert all(
        re.fullmatch(r"[a-z0-9][a-z0-9_.-]*==[0-9][a-zA-Z0-9_.+-]*", item)
        for item in dependencies
    )
    assert {item.partition("==")[0] for item in dependencies} == {
        "bandit",
        "cfn-lint",
        "pip-audit",
        "uv",
    }


def test_runtime_dependency_graph_and_base_image_are_immutable():
    pyproject = _read(Path("pyproject.toml"))
    lockfile = _read(Path("uv.lock"))
    dockerfile = _read(Path("Dockerfile"))
    assert 'required-version = "==0.11.33"' in pyproject
    assert 'torch = { index = "pytorch-cpu" }' in pyproject
    assert "version = 1" in lockfile
    assert re.search(
        r"PYTHON_BASE_IMAGE=python:3\.11\.15-slim@sha256:[0-9a-f]{64}",
        dockerfile,
    )
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "MODEL_ARTIFACT_SHA256" in _read(
        Path("scripts/prefetch_local_semantic_model.py")
    )


def test_ci_prefetches_revision_addressed_model_artifact_before_tests():
    workflow = _read(CI_WORKFLOW)
    assert "LOCAL_SEMANTIC_MODEL_CACHE: .model-cache" in workflow
    assert (
        "key: semantic-model-${{ runner.os }}-"
        "${{ env.LOCAL_SEMANTIC_MODEL_REVISION }}"
    ) in workflow
    assert workflow.index("Prefetch immutable semantic model") < workflow.index(
        "Run full suite with PostgreSQL integration"
    )


def test_dependabot_covers_runtime_container_and_actions():
    configuration = _read(DEPENDABOT)
    assert configuration.count("interval: weekly") == 3
    for ecosystem in ("pip", "docker", "github-actions"):
        assert f"package-ecosystem: {ecosystem}" in configuration


def test_runbook_does_not_claim_live_production_certification():
    runbook = " ".join(_read(RUNBOOK).lower().split())
    assert "not live-production certification" in runbook
    assert "does not claim live release closure" in runbook
    assert "do not point" in runbook
