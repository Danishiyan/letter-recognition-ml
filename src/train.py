"""Train, tune and compare all models on the same split.

Run:  python -m src.train
Outputs:
  reports/metrics.csv                      - one row per model
  reports/figures/model_comparison.png     - accuracy / macro-F1 bar chart
  reports/figures/confusion_matrix_best.png
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import RANDOM_STATE, get_train_test

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIGURES = REPORTS / "figures"

# name -> (estimator, small hyperparameter grid)
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), {"C": [0.1, 1, 10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {"max_depth": [10, 20, None]}),
    "kNN": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 5]}),
    "PCA (95%) + kNN": (
        Pipeline([("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                  ("knn", KNeighborsClassifier())]),
        {"knn__n_neighbors": [1, 3, 5]},
    ),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), {"n_estimators": [100, 300]}),
    "SVM (RBF)": (SVC(kernel="rbf"), {"C": [1, 10], "gamma": ["scale"]}),
    "MLP": (MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
            {"hidden_layer_sizes": [(64,), (128, 64)]}),
}


def main(models=MODELS):
    X_train, X_test, y_train, y_test = get_train_test()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rows, best = [], None
    for name, (estimator, grid) in models.items():
        start = time.time()
        search = GridSearchCV(estimator, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)            # tune on training data only
        y_pred = search.predict(X_test)         # evaluate once on unseen test data
        row = {
            "model": name,
            "best_params": search.best_params_,
            "cv_accuracy": search.best_score_,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_macro_f1": f1_score(y_test, y_pred, average="macro"),
            "train_seconds": time.time() - start,
        }
        rows.append(row)
        print(f"{name:20s} test acc = {row['test_accuracy']:.4f}  ({row['train_seconds']:.0f}s)")
        if best is None or row["test_accuracy"] > best[1]:
            best = (name, row["test_accuracy"], y_pred)

    results = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "metrics.csv", index=False)

    # Comparison chart
    ax = results.set_index("model")[["test_accuracy", "test_macro_f1"]].plot.barh(figsize=(9, 5))
    ax.set(title="Model comparison (held-out test set)", xlabel="Score", xlim=(0.5, 1.0))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close()

    # Confusion matrix of the best model
    name, acc, y_pred = best
    fig, ax = plt.subplots(figsize=(11, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False,
                                            include_values=False, cmap="Blues")
    ax.set_title(f"Confusion matrix - {name} (test accuracy {acc:.3f})")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix_best.png", dpi=150)
    plt.close()

    print("\n", results[["model", "test_accuracy", "test_macro_f1"]].round(4).to_string(index=False))
    return results


if __name__ == "__main__":
    main()
