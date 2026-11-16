"""Preprocessing: train/test split and feature scaling.

Order matters: we split FIRST, then fit the scaler on the training rows only.
Fitting on all rows would leak information about the test set into training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_data
from src.features import add_engineered_features

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_train_test(engineered: bool = True):
    """Return X_train, X_test, y_train, y_test (scaled) and the fitted scaler.

    The scaler is returned so it can be saved with the model: new data must be
    scaled with the SAME mean/std that the model was trained on.
    """
    df = load_data()
    if engineered:
        df = add_engineered_features(df)

    X = df.drop(columns="letter")
    y = df["letter"]

    # stratify=y keeps the same share of every letter in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)  # learn mean/std from TRAIN only
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_train_test()
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Train mean (should be ~0):", round(X_train.values.mean(), 4))
    print("Train std  (should be ~1):", round(X_train.values.std(), 4))
