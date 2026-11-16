"""Feature engineering: five new features built from the 16 originals.

Ported from my own feature-engineering notebook and cleaned up into a
reusable function so every later step (PCA, models) uses the same logic.
"""

import pandas as pd

ENGINEERED_FEATURES = [
    "box_area",        # width x height: overall size of the letter's box
    "aspect_ratio",    # width / height: wide letters (W, M) vs narrow (I, J)
    "pixel_density",   # onpix / box_area: how filled-in the box is
    "edge_ratio",      # x_edge / y_edge: horizontal vs vertical edges
    "total_variance",  # x2bar + y2bar: overall spread of "on" pixels
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the five engineered feature columns added.

    Ratios floor their denominator at 1 instead of adding a tiny epsilon:
    dividing by 0.000001 would create huge outliers for zero-height boxes.
    """
    out = df.copy()  # never modify the caller's DataFrame
    out["box_area"] = out["width"] * out["height"]
    out["aspect_ratio"] = out["width"] / out["height"].clip(lower=1)
    out["pixel_density"] = out["onpix"] / out["box_area"].clip(lower=1)
    out["edge_ratio"] = out["x_edge"] / out["y_edge"].clip(lower=1)
    out["total_variance"] = out["x2bar"] + out["y2bar"]
    return out


if __name__ == "__main__":
    from src.data import load_data

    df = add_engineered_features(load_data())
    print("Shape:", df.shape)
    print(df[["letter"] + ENGINEERED_FEATURES].head())
    print("\nRows where y_edge == 0 (floor used):", int((df["y_edge"] == 0).sum()))
