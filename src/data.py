# src/data.py
import pandas as pd
from sklearn import datasets
from typing import Tuple

def load_iris() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in X.columns]
    return X, y

def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load any CSV dataset from the given path.
    Example: load_from_csv("data/heart.csv")
    """
    df = pd.read_csv(path)
    return df

def load_heart_dataset():
    """
    Loads YOUR Kaggle heart dataset.
    Make sure heart.csv is inside: ds-ml-app/data/
    """
    df = pd.read_csv("data/heart.csv")
    return df
