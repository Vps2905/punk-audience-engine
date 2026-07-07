import json
import zipfile

import pandas as pd

from scripts.build_v2_review_package import main


def test_build_v2_review_package(monkeypatch, tmp_path):
    run_dir = tmp_path / "prompt_test"
    v2_dir = run_dir / "06_v2_autonomous_preview"
    embed_dir = v2_dir / "embeddings"
    embed_dir.mkdir(parents=True)

    final_summary = {
        "run_id": "prompt_test",
        "source_mode": "postgres_safe_derived",
        "source_rows": 216,
        "privacy_cohorts": 95,
        "v2_autonomous": {
            "status": "completed",
            "data_freshness": {
                "freshness_status": "fresh",
                "latest_source_timestamp": "2026-07-06T18:46:16+00:00",
                "source_rows_checked": 216,
            },
            "prompt_intent": {
                "business_intent": "restaurant+cafe",
                "confidence_score": 0.9,
                "locations": ["montreal", "san francisco"],
                "canonical_categories": ["restaurant", "cafe"],
                "dayparts": ["evening"],
            },
            "embedding_manifest": {
                "embedding_scope": "all_safe_cohorts",
                "embedding_backend": "sklearn_hashing",
                "vector_count": 95,
                "vector_dimension": 384,
            },
            "ranked_match_count": 95,
            "coverage_warnings": ["san francisco missing exact category/daypart"],
            "mutation": {"suggestion_count": 9},
            "approval_required": True,
            "downstream_export_enabled": False,
        },
        "v2_swarm_review": {
            "overall_review_status": "needs_human_review",
            "coverage_warning_count": 1,
            "data_gap_count": 2,
            "recommendations": ["Review mutation suggestions before exporting any audience."],
        },
    }

    (run_dir / "final_prompt_summary.json").write_text(json.dumps(final_summary))
    (v2_dir / "v2_swarm_review.json").write_text(json.dumps(final_summary["v2_swarm_review"]))
    (embed_dir / "all_safe_cohort_embedding_manifest.json").write_text(
        json.dumps(final_summary["v2_autonomous"]["embedding_manifest"])
    )

    pd.DataFrame(
        {
            "rank": [1],
            "location_name": ["montreal"],
            "primary_poi_type": ["restaurant"],
            "created_day_part": ["evening"],
            "final_match_score": [0.8],
            "session_id": ["should_not_be_included"],
        }
    ).to_csv(v2_dir / "ranked_audience_matches.csv", index=False)

    output_dir = tmp_path / "packages"

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_v2_review_package.py",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    package_dir = output_dir / "v2_review_package_prompt_test"
    zip_path = output_dir / "v2_review_package_prompt_test.zip"

    assert (package_dir / "V2_REVIEW_REPORT.md").exists()
    assert (package_dir / "PACKAGE_MANIFEST.json").exists()
    assert (package_dir / "safe_ranked_matches_top50.csv").exists()
    assert zip_path.exists()

    safe_ranked = pd.read_csv(package_dir / "safe_ranked_matches_top50.csv")
    assert "session_id" not in safe_ranked.columns

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    assert "V2_REVIEW_REPORT.md" in names
    assert "PACKAGE_MANIFEST.json" in names
