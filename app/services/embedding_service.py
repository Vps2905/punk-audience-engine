import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.vector_store_service import save_vector_store, similarity_search


PROCESSED_DIR = Path("data/processed")
VECTOR_DIR = Path("data/vectors")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def processed_path_for_job(job_id: str) -> Path:
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


def embed_processed_job(job_id: str) -> Dict[str, Any]:
    """
    Main embedding pipeline.

    1. Load processed feature CSV
    2. Convert each row to trait text
    3. Generate embeddings
    4. Save vectors + metadata
    """
    processed_path = processed_path_for_job(job_id)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    df = pd.read_csv(processed_path)

    if df.empty:
        raise ValueError("Processed feature table is empty. No rows to embed.")

    texts = dataframe_to_trait_texts(df)

    try:
        encoded = try_sentence_transformer_encode(texts)
    except Exception as e:
        encoded = tfidf_encode(job_id, texts)
        encoded["fallback_reason"] = str(e)

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
        model_info=model_info
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


def encode_query_for_job(job_id: str, query: str) -> np.ndarray:
    """
    Encodes search query using same backend used for the job.
    """
    model_path = VECTOR_DIR / f"{job_id}_model.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model info not found for job_id={job_id}")

    with open(model_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    backend = model_info.get("backend")

    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_info["model_name"])
        return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    if backend == "tfidf":
        vectorizer_path = model_info["vectorizer_path"]
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([query]).toarray()[0]

    raise ValueError(f"Unknown embedding backend: {backend}")


def search_similar_audiences(job_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Searches similar audience rows using vector similarity.
    """
    query_vector = encode_query_for_job(job_id, query)

    results = similarity_search(
        job_id=job_id,
        query_vector=query_vector,
        top_k=top_k
    )

    return {
        "job_id": job_id,
        "query": query,
        "top_k": top_k,
        "results": results,
    }
