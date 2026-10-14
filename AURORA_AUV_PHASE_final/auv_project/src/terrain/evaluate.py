"""
evaluate.py
-----------
Evaluation metrics and reporting for the terrain classifier, referenced by
`docs/architecture.md` (Stage 4: Terrain Classification).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.terrain.label_generation import CLASS_NAMES


@dataclass
class EvaluationResult:
    """Holds evaluation metrics for one model/split."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray
    report_text: str

    def __str__(self) -> str:
        class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
        header = f"{'':<24}" + "".join(f"{c[:10]:>12}" for c in class_labels)
        rows = []
        for i, row in enumerate(self.confusion):
            rows.append(f"{CLASS_NAMES[i]:<24}" + "".join(f"{v:>12}" for v in row))
        confusion_block = "\n".join(["Confusion matrix (rows=true, cols=predicted):", header] + rows)
        return (
            "Terrain Classifier Evaluation\n"
            "-------------------------------\n"
            f"Accuracy:     {self.accuracy:.4f}\n"
            f"Macro F1:     {self.macro_f1:.4f}\n"
            f"Weighted F1:  {self.weighted_f1:.4f}\n\n"
            f"{confusion_block}\n\n"
            f"{self.report_text}"
        )


def evaluate_classifier(model, X: np.ndarray, y_true: np.ndarray) -> EvaluationResult:
    """
    Evaluate a fitted classifier against a labeled split.

    Input:
        model: fitted classifier with .predict().
        X: (N, F) feature array.
        y_true: (N,) ground-truth (synthetic) labels.
    Processing:
        Computes accuracy, macro-F1 (unweighted mean across classes --
        the fair metric given class imbalance, so a good Obstacle score
        can't be masked by the majority Flat/Sand class), weighted-F1,
        the confusion matrix, and a per-class precision/recall/F1 report.
        All classes 0-3 are always included in the confusion matrix/report
        even if a class happens to be absent from this particular split.
    Output:
        EvaluationResult with all of the above.
    """
    y_pred = model.predict(X)
    labels = sorted(CLASS_NAMES)
    target_names = [CLASS_NAMES[i] for i in labels]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )

    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion=confusion,
        report_text=report_text,
    )


def feature_importance_report(model, feature_order=("depth", "slope", "roughness", "curvature")) -> str:
    """
    Render the RandomForest's feature_importances_ as a readable table.

    Input:
        model: fitted RandomForestClassifier.
        feature_order: names matching the column order used at training time.
    Processing:
        Pairs model.feature_importances_ with feature_order, sorts
        descending.
    Output:
        Formatted string, most-important feature first.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(feature_order, importances), key=lambda p: p[1], reverse=True)
    lines = ["Feature Importances", "---------------------"]
    for name, importance in pairs:
        lines.append(f"{name:<12}{importance:.4f}")
    return "\n".join(lines)
