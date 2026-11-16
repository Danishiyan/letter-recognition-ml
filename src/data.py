"""Download and load the UCI Letter Recognition dataset."""

from pathlib import Path
import urllib.request

import pandas as pd

# Where the original dataset lives online (UCI Machine Learning Repository)
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "letter-recognition/letter-recognition.data"
)

# Project paths, built from THIS file's location so they work from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "letter-recognition.data"

# The raw file has no header row, so we name the 17 columns ourselves
COLUMNS = [
    "letter",                      # target: A-Z
    "x_box", "y_box",              # position of the bounding box
    "width", "height",             # size of the bounding box
    "onpix",                       # number of "on" pixels
    "x_bar", "y_bar",              # mean x / y of on pixels
    "x2bar", "y2bar", "xybar",     # variances / correlation
    "x2ybar", "xy2bar",            # higher-order moments
    "x_edge", "x_edge_y",          # horizontal edge counts
    "y_edge", "y_edge_x",          # vertical edge counts
]


def download_data(force: bool = False) -> Path:
    """Download the raw data file once and return its path."""
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    return RAW_FILE


def load_data() -> pd.DataFrame:
    """Return the dataset as a pandas DataFrame (20,000 rows x 17 columns)."""
    path = download_data()
    return pd.read_csv(path, header=None, names=COLUMNS)


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
