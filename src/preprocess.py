# src/preprocess.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple

def train_test_split_scaled(
        X: pd.DataFrame,
        y: pd.Series,
        test_size=0.2,
        random_state=42
    ) -> Tuple:

    # Split data WITHOUT stratify (fixes ValueError)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
