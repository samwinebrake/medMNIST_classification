from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)


def compute_classification_metrics(y_true, y_pred, y_prob):
    """
    Compute standard multiclass classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like
        Predicted class probabilities, typically of shape
        `(n_samples, n_classes)`.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro F1 score, confusion matrix,
        and one-vs-rest AUC when available. If AUC cannot be computed,
        `auc_ovr` is set to `None`.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        metrics["auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        metrics["auc_ovr"] = None

    return metrics
