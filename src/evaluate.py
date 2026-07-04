"""
evaluate.py

Contains evaluation utilities for
classification models.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluates a classification model.

    Parameters
    ----------
    y_true : Actual labels

    y_pred : Predicted labels

    y_prob : Predicted probabilities

    Returns
    -------
    Dictionary containing all evaluation metrics.
    """

    results = {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred
        ),

        "Recall": recall_score(
            y_true,
            y_pred
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob
        ),

        "PR-AUC": average_precision_score(
            y_true,
            y_prob
        ),

        "Confusion Matrix": confusion_matrix(
            y_true,
            y_pred
        )

    }

    return results