"""
train.py
--------
Trains a Random Forest classifier on the preprocessed AI4I 2020 dataset
and saves the model to models/random_forest_model.joblib.

Run:
    python -m src.train
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src.preprocess import run_pipeline, FEATURE_COLS

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.joblib")


# ──────────────────────────────────────────────────────────────────────────
def build_model() -> RandomForestClassifier:
    """
    Return a configured RandomForestClassifier.

    class_weight='balanced' compensates for the heavy class imbalance
    (~97 % non-failure vs ~3 % failure).  Without it the model would
    learn to always predict 'no failure' and still score ~97 % accuracy —
    useless for predictive maintenance where missing a failure is costly.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,          # grow full trees; forest controls overfitting
        min_samples_leaf=2,
        class_weight="balanced", # upweight the minority failure class
        random_state=42,
        n_jobs=-1,               # use all CPU cores
    )


def evaluate(model, X_test, y_test) -> dict:
    """Compute and print all evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of predicted failures, how many were real)")
    print(f"  Recall    : {rec:.4f}  ← KEY metric: % of real failures caught")
    print(f"  F1-Score  : {f1:.4f}")
    print("="*55)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Failure", "Failure"]))

    # ── Confusion matrix plot ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Failure", "Failure"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"[train] Confusion matrix saved → {plot_path}")

    # ── Feature importance ─────────────────────────────────────────────────
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(range(len(FEATURE_COLS)),
            importances[indices],
            color="steelblue")
    ax2.set_xticks(range(len(FEATURE_COLS)))
    ax2.set_xticklabels([FEATURE_COLS[i] for i in indices],
                        rotation=35, ha="right", fontsize=9)
    ax2.set_title("Feature Importances — Random Forest")
    ax2.set_ylabel("Mean Decrease in Impurity")
    plt.tight_layout()
    fi_path = os.path.join(BASE_DIR, "models", "feature_importance.png")
    plt.savefig(fi_path, dpi=120)
    plt.close()
    print(f"[train] Feature importance plot saved → {fi_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[train] Model saved → {path}")


def train():
    """End-to-end training entry point."""
    # 1. Preprocess
    X_train, X_test, y_train, y_test = run_pipeline()

    # 2. Train
    print("\n[train] Training Random Forest …")
    model = build_model()
    model.fit(X_train, y_train)
    print("[train] Training complete.")

    # 3. Evaluate
    metrics = evaluate(model, X_test, y_test)

    # 4. Save
    save_model(model)

    return model, metrics


# ── Standalone run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
