from collections import defaultdict
from pathlib import Path


def test_sql_migration_numeric_prefixes_are_unique():
    migrations_by_prefix: dict[str, list[str]] = defaultdict(list)

    for migration in sorted(Path("migrations").glob("*.sql")):
        prefix, separator, _name = migration.name.partition("_")
        if separator and prefix.isdigit():
            migrations_by_prefix[prefix].append(migration.name)

    duplicates = {
        prefix: names
        for prefix, names in migrations_by_prefix.items()
        if len(names) > 1
    }

    assert duplicates == {}
