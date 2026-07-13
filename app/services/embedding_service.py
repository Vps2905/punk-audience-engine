import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from app.services.vector_store_service import (
    load_vector_store,
    save_vector_store,
    similarity_search,
)
from app.services.postgres_vector_store_service import (
    load_postgres_vector_model_info,
)
from app.core.production_guardrails import require_local_file_storage_allowed


PROCESSED_DIR = Path("data/processed")
VECTOR_DIR = Path("data/vectors")


_QUERY_ONTOLOGY_ALIASES = {
    "coffee shop": (
        "cafe",
        "primary_poi_type cafe",
    ),
    "coffeehouse": (
        "cafe",
        "primary_poi_type cafe",
    ),
    "espresso bar": (
        "cafe",
        "primary_poi_type cafe",
    ),
    "co working": (
        "coworking_space",
        "coworking space",
    ),
    "coworking": (
        "coworking_space",
        "coworking space",
    ),
    "fitness center": (
        "gym",
        "primary_poi_type gym",
    ),
    "fitness centre": (
        "gym",
        "primary_poi_type gym",
    ),
    "supermarket": (
        "grocery_store",
        "grocery store",
    ),
    "barbershop": (
        "barber_shop",
        "barber shop",
    ),
}


def normalize_embedding_query(query: str) -> str:
    """
    Enrich user language with normalized audience taxonomy
    terms used by stored embedding metadata.
    """
    original = " ".join(
        str(query or "").strip().split()
    )

    if not original:
        raise ValueError("query cannot be empty.")

    searchable = unicodedata.normalize(
        "NFKD",
        original,
    )
    searchable = searchable.encode(
        "ascii",
        "ignore",
    ).decode("ascii").lower()

    searchable = re.sub(
        r"[^a-z0-9]+",
        " ",
        searchable,
    )
    searchable = " ".join(searchable.split())

    additions = []

    for alias, canonical_terms in (
        _QUERY_ONTOLOGY_ALIASES.items()
    ):
        if alias not in searchable:
            continue

        for canonical in canonical_terms:
            canonical_search = canonical.replace(
                "_",
                " ",
            ).lower()

            if (
                canonical_search not in searchable
                and canonical not in additions
            ):
                additions.append(canonical)

    if not additions:
        return original

    return " ".join(
        [original, *additions]
    )


def processed_path_for_job(job_id: str) -> Path:
    require_local_file_storage_allowed("local processed feature CSV input")
    return PROCESSED_DIR / f"{job_id}_clean_features.csv"


def build_trait_text(row: pd.Series) -> str:
    """
    Converts one processed audience row into natural-language-like text.

    Example:
    city Hyderabad interest fitness visit_time evening affinity high age_group 25_34
    """
    parts = []

    for col, value in row.items():
        if pd.isna(value):
            continue

        if col in ["privacy_status", "k_min"]:
            continue

        parts.append(f"{col} {value}")

    return " ".join(parts)


def dataframe_to_trait_texts(df: pd.DataFrame) -> List[str]:
    """
    Converts all processed feature rows into trait text.
    """
    return [build_trait_text(row) for _, row in df.iterrows()]


def _embedding_backend() -> str:
    return os.getenv(
        "EMBEDDING_BACKEND",
        "auto",
    ).strip().lower()


def _use_postgres_vector_backend() -> bool:
    return os.getenv(
        "VECTOR_BACKEND",
        "local",
    ).strip().lower() in {
        "postgres",
        "postgres_array",
        "pg_array",
        "pgvector",
    }


def hashing_encode(texts: List[str], n_features: int | None = None) -> Dict[str, Any]:
    """
    Production-safe stateless embedding fallback.

    Uses sklearn HashingVectorizer so query-time encoding can be recreated
    without writing a local vectorizer artifact.
    """
    dimension = int(n_features or os.getenv("EMBEDDING_HASHING_FEATURES", "384"))
    vectorizer = HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
    )
    vectors = vectorizer.transform(texts).toarray()

    return {
        "vectors": vectors,
        "backend": "sklearn_hashing",
        "model_name": "sklearn_hashing_vectorizer",
        "dimension": dimension,
    }


def try_sentence_transformer_encode(texts: List[str]) -> Dict[str, Any]:
    """
    Tries to use sentence-transformers for semantic embeddings.

    If model is not available or download fails, caller will use fallback.
    """
    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return {
        "vectors": vectors,
        "backend": "sentence-transformers",
        "model_name": model_name,
    }


def tfidf_encode(job_id: str, texts: List[str]) -> Dict[str, Any]:
    """
    Fallback embedding method using TF-IDF.

    This is not as semantic as sentence-transformers,
    but it is reliable and local.
    """
    require_local_file_storage_allowed("local TF-IDF vectorizer artifact")
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()

    vectorizer_path = VECTOR_DIR / f"{job_id}_tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)

    return {
        "vectors": vectors,
        "backend": "tfidf",
        "model_name": "sklearn_tfidf",
        "vectorizer_path": str(vectorizer_path),
    }


def _encode_texts_for_job(job_id: str, texts: List[str]) -> Dict[str, Any]:
    requested_backend = _embedding_backend()

    if requested_backend in {"sklearn_hashing", "hashing"}:
        return hashing_encode(texts)

    if requested_backend in {"sentence-transformers", "sentence_transformers"}:
        return try_sentence_transformer_encode(texts)

    if requested_backend == "tfidf":
        return tfidf_encode(job_id, texts)

    if requested_backend not in {"", "auto"}:
        raise ValueError(
            "Unsupported EMBEDDING_BACKEND. Use auto, sklearn_hashing, "
            "sentence-transformers, or tfidf."
        )

    try:
        return try_sentence_transformer_encode(texts)
    except Exception as e:
        vector_backend = os.getenv("VECTOR_BACKEND", "local").strip().lower()
        if vector_backend in {"postgres", "postgres_array", "pg_array", "pgvector"}:
            encoded = hashing_encode(texts)
        else:
            encoded = tfidf_encode(job_id, texts)
        encoded["fallback_reason"] = str(e)
        return encoded


def embed_dataframe(job_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Embeds a safe feature dataframe and stores vectors through the active vector backend.
    """
    if df.empty:
        raise ValueError("Processed feature table is empty. No rows to embed.")

    texts = dataframe_to_trait_texts(df)
    encoded = _encode_texts_for_job(job_id, texts)
    vectors = np.array(encoded["vectors"])

    metadata = []
    for idx, row in df.iterrows():
        record = row.to_dict()
        record["trait_text"] = texts[idx]
        record["vector_index"] = int(idx)
        metadata.append(record)

    model_info = {
        "backend": encoded["backend"],
        "model_name": encoded["model_name"],
        "vector_count": int(len(vectors)),
        "dimension": int(vectors.shape[1]) if len(vectors.shape) > 1 else 0,
    }

    if "vectorizer_path" in encoded:
        model_info["vectorizer_path"] = encoded["vectorizer_path"]

    if "fallback_reason" in encoded:
        model_info["fallback_reason"] = encoded["fallback_reason"]

    saved_paths = save_vector_store(
        job_id=job_id,
        vectors=vectors,
        metadata=metadata,
        model_info=model_info,
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Embeddings generated and stored",
        "rows_embedded": len(metadata),
        "embedding_backend": model_info["backend"],
        "model_name": model_info["model_name"],
        "dimension": model_info["dimension"],
        **saved_paths,
    }


def embed_records(job_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Production-safe embedding entry point.

    Accepts already-safe cohort/feature records directly instead of requiring
    a local processed CSV file.
    """
    if not records:
        raise ValueError("records cannot be empty.")

    df = pd.DataFrame(records)
    return embed_dataframe(job_id=job_id, df=df)


def embed_processed_job(job_id: str) -> Dict[str, Any]:
    """
    Legacy/dev embedding pipeline.

    Loads local processed feature CSV, then stores vectors through the active
    vector backend. In production, prefer embed_records to avoid local file input.
    """
    processed_path = processed_path_for_job(job_id)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)
    return embed_dataframe(job_id=job_id, df=df)


def encode_query_for_job(job_id: str, query: str) -> np.ndarray:
    """
    Encodes search query using same backend used for the job.
    """
    normalized_query = normalize_embedding_query(
        query
    )

    if _use_postgres_vector_backend():
        model_info = (
            load_postgres_vector_model_info(
                job_id
            )
        )
    else:
        store = load_vector_store(job_id)
        model_info = store["model_info"]

    backend = model_info.get("backend")

    if backend in {"sklearn_hashing", "hashing"}:
        dimension = int(
            model_info.get("dimension")
            or model_info.get("vector_dimension")
            or os.getenv("EMBEDDING_HASHING_FEATURES", "384")
        )
        return hashing_encode([normalized_query], n_features=dimension)["vectors"][0]

    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_info["model_name"])
        return model.encode([normalized_query], convert_to_numpy=True, normalize_embeddings=True)[0]

    if backend == "tfidf":
        vectorizer_path = model_info["vectorizer_path"]
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([normalized_query]).toarray()[0]

    raise ValueError(f"Unknown embedding backend: {backend}")


def search_similar_audiences(
    job_id: str,
    query: str,
    top_k: int = 5,
    *,
    location_name: str | None = None,
    primary_poi_type: str | None = None,
    created_day_part: str | None = None,
    min_quality: float | None = None,
) -> Dict[str, Any]:
    """
    Search privacy-safe audience vectors with optional
    structured metadata filters.
    """
    normalized_query = normalize_embedding_query(
        query
    )

    query_vector = encode_query_for_job(
        job_id,
        normalized_query,
    )

    results = similarity_search(
        job_id=job_id,
        query_vector=query_vector,
        top_k=top_k,
        location_name=location_name,
        primary_poi_type=primary_poi_type,
        created_day_part=created_day_part,
        min_quality=min_quality,
    )

    return {
        "job_id": job_id,
        "query": query,
        "normalized_query": normalized_query,
        "top_k": top_k,
        "filters": {
            "location_name": location_name,
            "primary_poi_type": primary_poi_type,
            "created_day_part": created_day_part,
            "min_quality": min_quality,
        },
        "results": results,
    }
