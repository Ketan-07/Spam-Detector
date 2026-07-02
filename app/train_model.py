"""
Trains the TF-IDF + Logistic Regression spam classifier and saves it
as a single pipeline artifact (model/spam_model.pkl) that the API loads
at request time.

Run from the project root:
    python app/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, encoding_errors="ignore")

    x = df["v2"]
    y = df["v1"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    print("Training pipeline (TF-IDF + Logistic Regression) ...")
    pipe.fit(xtrain, ytrain)

    print("\nTrain performance:")
    print(classification_report(ytrain, pipe.predict(xtrain)))

    print("Test performance:")
    print(classification_report(ytest, pipe.predict(xtest)))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
