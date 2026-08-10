from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from urllib.parse import quote


_HOST_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")
_DATABASE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def _required(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the production runtime.")
    return value


def build_database_url(environment: Mapping[str, str]) -> str:
    host = _required(environment, "DATABASE_HOST")
    database = _required(environment, "DATABASE_NAME")
    username = _required(environment, "DATABASE_USER")
    password = _required(environment, "DATABASE_PASSWORD")
    if not _HOST_PATTERN.fullmatch(host):
        raise RuntimeError("DATABASE_HOST is invalid.")
    if not _DATABASE_PATTERN.fullmatch(database):
        raise RuntimeError("DATABASE_NAME is invalid.")
    raw_port = str(environment.get("DATABASE_PORT") or "5432").strip()
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise RuntimeError("DATABASE_PORT is invalid.") from exc
    if not 1 <= port <= 65_535:
        raise RuntimeError("DATABASE_PORT is invalid.")
    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return (
        f"postgresql://{encoded_user}:{encoded_password}@"
        f"{host}:{port}/{database}?sslmode=require"
    )


def configured_environment(
    mode: str,
    environment: Mapping[str, str],
) -> dict[str, str]:
    if mode not in {"api", "worker", "migration"}:
        raise RuntimeError("Runtime mode must be api, worker, or migration.")
    result = dict(environment)
    database_url = build_database_url(environment)
    target = {
        "api": "ECHO_DATABASE_URL",
        "worker": "PROVIDER_INGESTION_DATABASE_URL",
        "migration": "DATABASE_URL",
    }[mode]
    result[target] = database_url
    result.pop("DATABASE_USER", None)
    result.pop("DATABASE_PASSWORD", None)
    return result


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: production_runtime_entrypoint.py <api|worker|migration>"
        )
    mode = sys.argv[1]
    environment = configured_environment(mode, os.environ)
    command = {
        "api": [
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        "worker": ["python", "scripts/run_provider_ingestion_worker.py"],
        "migration": ["python", "scripts/bootstrap_preproduction_database.py"],
    }.get(mode)
    if command is None:
        raise RuntimeError("Runtime mode must be api, worker, or migration.")
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
