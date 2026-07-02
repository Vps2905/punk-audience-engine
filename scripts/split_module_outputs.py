from pathlib import Path
import json
import shutil
import zipfile
from datetime import datetime, timezone


MODULES = [
    ("01_ingestion_privacy", "MODULE_1_INGESTION_PRIVACY.zip", "Module 1 - Ingestion & Privacy Layer"),
    ("02_embeddings_feature_store", "MODULE_2_EMBEDDINGS_FEATURE_STORE.zip", "Module 2 - Embedding & Feature Store"),
    ("03_cohort_management", "MODULE_3_COHORT_MANAGEMENT.zip", "Module 3 - Cohort Management & Lookalike"),
    ("04_meta_safe_export", "MODULE_4_META_SAFE_EXPORT.zip", "Module 4 - Meta Safe Export"),
    ("05_simple_conversational_trigger", "MODULE_5_CONVERSATIONAL_TRIGGER.zip", "Module 5 - Simple Conversational Trigger"),
]


def zip_dir(folder: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file in folder.rglob("*"):
            if file.is_file():
                z.write(file, file.relative_to(folder.parent))


def main() -> None:
    packages_root = Path("data/review_packages")
    packages = sorted(packages_root.glob("five_module_demo_*"), key=lambda p: p.stat().st_mtime, reverse=True)

    packages = [p for p in packages if p.is_dir()]

    if not packages:
        raise SystemExit("No five_module_demo_* package found. Run generate_five_module_demo_outputs.py first.")

    latest_package = packages[0]
    split_dir = latest_package / "separate_module_zips"
    split_dir.mkdir(exist_ok=True)

    created = []

    for folder_name, zip_name, title in MODULES:
        module_folder = latest_package / folder_name

        if not module_folder.exists():
            raise FileNotFoundError(f"Missing module folder: {module_folder}")

        readme = module_folder / "README.txt"
        if not readme.exists():
            readme.write_text(
                f"{title}\n\n"
                f"This folder contains separate output files for {title}.\n"
                "All files are generated from the safe demo pipeline.\n"
                "No raw MAIDs, raw observations, raw lat/lng, email, phone, or DB credentials are included.\n"
            )

        zip_path = split_dir / zip_name
        zip_dir(module_folder, zip_path)

        created.append({
            "module": title,
            "folder": str(module_folder),
            "zip": str(zip_path),
            "files": sorted([str(f.relative_to(module_folder)) for f in module_folder.rglob("*") if f.is_file()])
        })

    combined_zip = split_dir / "ALL_MODULES_COMBINED.zip"
    zip_dir(latest_package, combined_zip)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_package": str(latest_package),
        "separate_module_zip_folder": str(split_dir),
        "combined_zip": str(combined_zip),
        "modules": created,
        "privacy_note": "Separate module outputs do not include raw MAIDs, raw observations, raw lat/lng, email, phone, DB credentials, or individual-level device/user records."
    }

    summary_path = split_dir / "MASTER_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("DONE")
    print("Source package:", latest_package)
    print("Separate module ZIP folder:", split_dir)
    print("Created ZIPs:")
    for item in created:
        print("-", item["zip"])
    print("-" , combined_zip)
    print("Master summary:", summary_path)


if __name__ == "__main__":
    main()
