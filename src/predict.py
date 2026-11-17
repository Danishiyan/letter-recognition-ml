"""Load the saved model and predict letters for new data.

Run:  python -m src.predict
      (needs models/best_model.joblib - create it with: python -m src.train)

The demo takes 5 rows from the TEST set (data the model never trained on),
predicts their letters and compares with the true answers.
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import add_engineered_features
from src.preprocess import RANDOM_STATE, TEST_SIZE
from src.train import MODEL_FILE


def load_model(path=MODEL_FILE):
    """Load the saved bundle (pipeline + info). Fail with a clear message if missing."""
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run: python -m src.train")
    return joblib.load(path)


def predict_letters(raw_rows: pd.DataFrame, bundle=None) -> list:
    """Predict letters for rows with the 16 original feature columns.

    Same steps as training: add engineered features -> scale -> model.
    """
    bundle = bundle or load_model()
    rows = add_engineered_features(raw_rows)
    X = rows[bundle["features"]]          # same columns, same order as training
    return list(bundle["pipeline"].predict(X))


def main(n=10):
    bundle = load_model()
    print(f"Model: {bundle['model_name']} (test accuracy {bundle['test_accuracy']:.3f})\n")

    # Re-create the same split as training to get unseen test rows
    df = load_data()
    _, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["letter"],
                               random_state=RANDOM_STATE)
    sample = test.sample(n, random_state=RANDOM_STATE)

    predictions = predict_letters(sample.drop(columns="letter"), bundle)
    for true, pred in zip(sample["letter"], predictions):
        mark = "OK " if true == pred else "BAD"
        print(f"  true = {true}   predicted = {pred}   {mark}")


if __name__ == "__main__":
    main()
