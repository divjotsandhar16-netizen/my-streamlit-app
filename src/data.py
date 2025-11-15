# src/data.py
from sklearn import datasets
import pandas as pd
from typing import Tuple

def load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    # convert to nice column names
    X.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in X.columns]
    return X, y

def load_from_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    return df
