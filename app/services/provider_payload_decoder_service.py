from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List


class ProviderPayloadDecodeError(ValueError):
    def __init__(self, reason_code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.reason_code = reason_code
        self.safe_message = safe_message


class ProviderPayloadDecoderService:
    def decode(
        self,
        payload: bytes,
        *,
        data_format: str,
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        try:
            text_payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderPayloadDecodeError(
                "invalid_text_encoding",
                "The provider payload must use UTF-8 encoding.",
            ) from exc

        normalized_format = str(data_format or "").strip().lower()
        if normalized_format == "csv":
            rows = self._decode_csv(text_payload, max_rows=max_rows)
        elif normalized_format == "jsonl":
            rows = self._decode_jsonl(text_payload, max_rows=max_rows)
        else:
            raise ProviderPayloadDecodeError(
                "unsupported_data_format",
                "The provider payload format is not supported.",
            )

        if not rows:
            raise ProviderPayloadDecodeError(
                "payload_has_no_rows",
                "The provider payload did not contain any data rows.",
            )
        return rows

    def _decode_csv(
        self,
        payload: str,
        *,
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        stream = io.StringIO(payload, newline="")
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        normalized_fields = [str(value or "").strip() for value in fieldnames]
        if not normalized_fields or any(not value for value in normalized_fields):
            raise ProviderPayloadDecodeError(
                "invalid_csv_header",
                "The provider CSV must contain a non-empty header.",
            )
        if len(set(normalized_fields)) != len(normalized_fields):
            raise ProviderPayloadDecodeError(
                "duplicate_columns",
                "The provider payload contains duplicate column names.",
            )
        reader.fieldnames = normalized_fields

        rows: List[Dict[str, Any]] = []
        try:
            for row in reader:
                if None in row:
                    raise ProviderPayloadDecodeError(
                        "invalid_csv_row",
                        "A provider CSV row contains more values than the header.",
                    )
                rows.append({str(key): value for key, value in row.items()})
                self._enforce_row_limit(len(rows), max_rows)
        except csv.Error as exc:
            raise ProviderPayloadDecodeError(
                "invalid_csv",
                "The provider CSV could not be parsed.",
            ) from exc
        return rows

    def _decode_jsonl(
        self,
        payload: str,
        *,
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ProviderPayloadDecodeError(
                    "invalid_jsonl",
                    f"The provider JSONL contains invalid JSON on line {line_number}.",
                ) from exc
            if not isinstance(value, dict):
                raise ProviderPayloadDecodeError(
                    "invalid_jsonl_record",
                    "Every provider JSONL record must be an object.",
                )
            rows.append({str(key): item for key, item in value.items()})
            self._enforce_row_limit(len(rows), max_rows)
        return rows

    def _enforce_row_limit(self, row_count: int, max_rows: int) -> None:
        if row_count > max_rows:
            raise ProviderPayloadDecodeError(
                "row_limit_exceeded",
                "The provider payload exceeds the configured maximum row count.",
            )
