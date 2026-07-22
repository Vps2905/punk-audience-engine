import numpy as np
import pandas as pd
from typing import Any
from pydantic import BaseModel

def make_serializable(obj: Any) -> Any:
    """
    Recursively converts non-serializable objects (like numpy types and pandas NAs)
    into standard Python types for JSON or msgpack serialization.
    """
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, BaseModel):
        return make_serializable(obj.model_dump())
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj
