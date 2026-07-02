import hashlib


def sha256_hash(value: str) -> str:
    if value is None:
        return ""

    normalized = str(value).strip().lower()

    if normalized == "" or normalized == "nan":
        return ""

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
