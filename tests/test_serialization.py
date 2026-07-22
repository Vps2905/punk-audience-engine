import pytest
import numpy as np
import pandas as pd
from app.utils.serialization import make_serializable

def test_make_serializable():
    data = {
        "float": np.float64(3.14),
        "int": np.int64(42),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3], dtype=np.int32),
        "nested": {
            "val": np.float32(1.5),
            "nan": pd.NA,
            "nan2": np.nan,
            "list": [np.float64(1.0), 2]
        },
        "tuple": (np.int32(1), "test"),
        "set": {np.float64(2.0)}
    }
    
    clean = make_serializable(data)
    assert isinstance(clean["float"], float)
    assert isinstance(clean["int"], int)
    assert isinstance(clean["bool"], bool)
    assert isinstance(clean["array"], list)
    assert isinstance(clean["nested"]["val"], float)
    assert clean["nested"]["nan"] is None
    assert clean["nested"]["nan2"] is None
    assert isinstance(clean["nested"]["list"][0], float)
    assert isinstance(clean["tuple"], tuple)
    assert isinstance(clean["tuple"][0], int)
    assert isinstance(clean["set"], list)
    assert isinstance(clean["set"][0], float)
