import pytest

from app.services.provider_geography_normalization_service import (
    ProviderGeographyNormalizationService,
)


def test_accent_and_case_variants_share_stable_non_coordinate_geo_id():
    service = ProviderGeographyNormalizationService()

    first = service.normalize("Montréal")
    second = service.normalize("  MONTREAL  ")

    assert first.geo_id == second.geo_id
    assert first.canonical_name == "montreal"
    assert first.resolution_status == "unscoped_name_only_review_required"
    assert "latitude" not in first.to_safe_dict()
    assert "longitude" not in first.to_safe_dict()


def test_admin_scoped_names_do_not_collide_across_jurisdictions():
    service = ProviderGeographyNormalizationService()

    canada = service.normalize(
        "Springfield", country_code="ca", admin1_code="on"
    )
    united_states = service.normalize(
        "Springfield", country_code="us", admin1_code="il"
    )

    assert canada.geo_id != united_states.geo_id
    assert canada.resolution_status == "admin_scoped"


def test_invalid_or_incomplete_geography_fails_closed():
    service = ProviderGeographyNormalizationService()

    with pytest.raises(ValueError):
        service.normalize("unknown")
    with pytest.raises(ValueError, match="requires country"):
        service.normalize("Montreal", admin1_code="qc")
