# train.py
import argparse
import pandas as pd
from src.preprocess import train_test_split_scaled
from src.model import (
    build_default_model, train_model, evaluate_model, save_model
)

def main(output="model.joblib"):
    df = pd.read_csv("data/heart.csv")

    X = df.drop("target", axis=1)
    y = df["target"]
    columns = X.columns    # ADDED

    X_train, X_test, y_train, y_test, scaler = train_test_split_scaled(X, y)

    model = build_default_model()
    model = train_model(model, X_train, y_train)

    eval_res = evaluate_model(model, X_test, y_test)

    print("\n📌 Accuracy:", eval_res["accuracy"])
    print("\n📌 Classification Report:\n", eval_res["report"])

    # Now saves columns also
    save_model(model, scaler, columns, output)

    print("\n🔥 Saved trained model to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output model path", default="model.joblib")
    args = parser.parse_args()

    main(args.out)
