from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresSourceAgent:
    """
    Reads schema/table samples from Postgres using read-only access.

    No credentials should be hardcoded in committed code.
    Use local .env:
    ECHO_DATABASE_URL=postgresql://user:password@host:port/db?sslmode=prefer
    """

    def _connection_url(self, database_url: Optional[str] = None) -> str:
        url = (
            database_url
            or os.getenv("ECHO_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
        )

        if not url:
            raise ValueError(
                "No Postgres connection URL found. Set ECHO_DATABASE_URL in .env or environment."
            )

        return url

    def _validate_identifier(self, value: str, label: str) -> str:
        if not SAFE_IDENTIFIER.match(value):
            raise ValueError(f"Unsafe {label}: {value}")
        return value

    def _engine(self, database_url: Optional[str] = None):
        from sqlalchemy import create_engine
        return create_engine(self._connection_url(database_url))

    def list_tables(
        self,
        schema_name: str = "public",
        database_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        engine = self._engine(database_url)

        query = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %(schema_name)s
        ORDER BY table_name;
        """

        df = pd.read_sql_query(query, engine, params={"schema_name": schema_name})

        return {
            "agent": "postgres_source_agent",
            "status": "tables_listed",
            "schema_name": schema_name,
            "table_count": len(df),
            "tables": df.to_dict(orient="records"),
        }

    def sample_table(
        self,
        schema_name: str,
        table_name: str,
        limit: int = 10000,
        database_url: Optional[str] = None,
    ) -> pd.DataFrame:
        schema_name = self._validate_identifier(schema_name, "schema_name")
        table_name = self._validate_identifier(table_name, "table_name")
        limit = max(1, min(int(limit), 100000))

        engine = self._engine(database_url)

        query = f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {limit};'
        return pd.read_sql_query(query, engine)

    def table_row_count(
        self,
        schema_name: str,
        table_name: str,
        database_url: Optional[str] = None,
    ) -> int:
        schema_name = self._validate_identifier(schema_name, "schema_name")
        table_name = self._validate_identifier(table_name, "table_name")

        engine = self._engine(database_url)

        query = f'SELECT COUNT(*) AS row_count FROM "{schema_name}"."{table_name}";'
        df = pd.read_sql_query(query, engine)

        return int(df.iloc[0]["row_count"])
