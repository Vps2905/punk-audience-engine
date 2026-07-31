from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class CanonicalProviderGeography:
    geo_id: str
    canonical_name: str
    country_code: str | None
    admin1_code: str | None
    resolution_status: str
    normalization_version: str = "provider-geo-v1"

    def to_safe_dict(self) -> dict[str, str | None]:
        return {
            "geo_id": self.geo_id,
            "canonical_name": self.canonical_name,
            "country_code": self.country_code,
            "admin1_code": self.admin1_code,
            "geo_resolution_status": self.resolution_status,
            "geo_normalization_version": self.normalization_version,
        }


class ProviderGeographyNormalizationService:
    """Create stable geo keys without retaining exact coordinates."""

    def normalize(
        self,
        location_name: object,
        *,
        country_code: object = None,
        admin1_code: object = None,
    ) -> CanonicalProviderGeography:
        canonical_name = self._text(location_name)
        if not canonical_name or canonical_name == "unknown":
            raise ValueError("A supported location name is required")
        country = self._code(country_code, "country_code")
        admin1 = self._code(admin1_code, "admin1_code")
        if admin1 and not country:
            raise ValueError("admin1_code requires country_code")

        if country and admin1:
            scope = f"{country}:{admin1}"
            status = "admin_scoped"
        elif country:
            scope = f"{country}:_"
            status = "country_scoped_review_required"
        else:
            scope = "unscoped"
            status = "unscoped_name_only_review_required"

        digest = hashlib.sha256(
            f"provider-geo-v1\x1f{scope}\x1f{canonical_name}".encode("utf-8")
        ).hexdigest()[:24]
        return CanonicalProviderGeography(
            geo_id=f"geo_{digest}",
            canonical_name=canonical_name,
            country_code=country,
            admin1_code=admin1,
            resolution_status=status,
        )

    def _text(self, value: object) -> str:
        decomposed = unicodedata.normalize("NFKD", str(value or ""))
        ascii_text = "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        )
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            ascii_text.casefold(),
        )
        return " ".join(normalized.split())[:256]

    def _code(self, value: object, field_name: str) -> str | None:
        if value is None or not str(value).strip():
            return None
        normalized = str(value).strip().casefold()
        if not _CODE_RE.fullmatch(normalized):
            raise ValueError(f"{field_name} is invalid")
        return normalized
