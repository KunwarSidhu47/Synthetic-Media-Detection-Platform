"""
Model evaluation metrics utility functions.
"""

from typing import List, Union, Optional
import numpy as np

from backend.schemas.detection import EvaluationMetrics


def compute_evaluation_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[float], np.ndarray]] = None
) -> EvaluationMetrics:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
    
    Args:
        y_true: Ground truth binary labels (0 = REAL, 1 = SYNTHETIC).
        y_pred: Predicted binary labels (0 = REAL, 1 = SYNTHETIC).
        y_prob: Optional predicted probability scores for synthetic class [0.0, 1.0].
        
    Returns:
        EvaluationMetrics schema object.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, np.array(y_prob)))
        except ValueError:
            roc_auc = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return EvaluationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=roc_auc,
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    )
