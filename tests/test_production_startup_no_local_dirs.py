import os
import subprocess
import sys
from pathlib import Path


def test_production_app_import_creates_no_local_data_dirs(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]

    # StaticFiles uses relative app/static paths during app import.
    # The symlink gives the isolated subprocess the real application
    # package while keeping all runtime paths under tmp_path.
    (tmp_path / "app").symlink_to(
        repo_root / "app",
        target_is_directory=True,
    )

    script = """
from pathlib import Path
import app.main  # noqa: F401

data_path = Path("data")
if data_path.exists():
    created = sorted(str(path) for path in data_path.rglob("*"))
    raise RuntimeError(
        "Production app import created local data paths: "
        + repr(created)
    )
"""

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(tmp_path),
            "PRODUCTION_MODE": "true",
            "APP_ENV": "production",
            "ALLOW_LOCAL_FILE_STORAGE": "false",
            "ALLOW_DEMO_ROUTES": "false",
            "AUDIENCE_JOB_STORE_BACKEND": "postgres",
            "VECTOR_BACKEND": "postgres",
            "EMBEDDING_BACKEND": "sklearn_hashing",
            "ALL_SAFE_COHORT_EMBEDDING_STORE": "postgres",
            "ECHO_DATABASE_URL": (
                "postgresql://user:pass@localhost/test"
            ),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        result.stdout + "\n" + result.stderr
    )
    assert not (tmp_path / "data").exists()
