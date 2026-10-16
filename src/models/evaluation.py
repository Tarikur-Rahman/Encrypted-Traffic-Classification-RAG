from __future__ import annotations
import math
from typing import Mapping, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

def calibration(y_true, y_pred, confidence, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    correct = (y_true == y_pred).astype(float)
    rows, ece = [], 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (confidence >= lo) & (confidence <= hi if i == bins-1 else confidence < hi)
        n = int(mask.sum())
        if n:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            gap = abs(acc-conf)
            ece += n/len(y_true)*gap
        else:
            acc = conf = gap = float("nan")
        rows.append({"bin_id": i, "lower": lo, "upper": hi, "count": n,
                     "accuracy": acc, "mean_confidence": conf, "absolute_gap": gap})
    return float(ece), rows

def evaluate(y_true, probabilities, class_ids: Sequence[int], class_names: Mapping[int,str] | None=None):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    class_ids = [int(x) for x in class_ids]
    if p.shape != (len(y_true), len(class_ids)):
        raise ValueError("Probability matrix shape mismatch")
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid probabilities")
    argmax = p.argmax(axis=1)
    y_pred = np.array([class_ids[i] for i in argmax], dtype=int)
    conf = p[np.arange(len(y_true)), argmax]

    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
        "macro_recall": float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
        "macro_f1": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "weighted_f1": float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true,y_pred)),
    }
    onehot = label_binarize(y_true, classes=class_ids)
    if len(class_ids) == 2 and onehot.shape[1] == 1:
        onehot = np.column_stack([1-onehot[:,0], onehot[:,0]])
    try:
        overall["auroc_macro_ovr"] = float(roc_auc_score(onehot, p, average="macro", multi_class="ovr"))
    except ValueError:
        overall["auroc_macro_ovr"] = None
    try:
        overall["auprc_macro"] = float(average_precision_score(onehot, p, average="macro"))
    except ValueError:
        overall["auprc_macro"] = None
    yoh = np.zeros_like(p)
    mapping = {c:i for i,c in enumerate(class_ids)}
    for i,y in enumerate(y_true):
        yoh[i,mapping[int(y)]] = 1
    overall["brier_score"] = float(np.mean(np.sum((p-yoh)**2, axis=1)))
    overall["ece"], cal = calibration(y_true, y_pred, conf)

    per_class = [
        {"class_id": cid, "class_name": (class_names or {}).get(cid,str(cid)),
         "precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i,cid in enumerate(class_ids)
    ]
    cm = confusion_matrix(y_true,y_pred,labels=class_ids)
    cm_long = [
        {"actual_class_id": a, "predicted_class_id": b, "count": int(cm[i,j])}
        for i,a in enumerate(class_ids) for j,b in enumerate(class_ids)
    ]
    preds = []
    for i in range(len(y_true)):
        row = {"sample_index": i, "y_true": int(y_true[i]), "y_pred": int(y_pred[i]),
               "confidence": float(conf[i]), "correct": int(y_true[i]==y_pred[i])}
        for j,c in enumerate(class_ids):
            row[f"p_class_{c}"] = float(p[i,j])
        preds.append(row)
    return {"overall":overall,"per_class":per_class,"confusion_matrix_long":cm_long,
            "calibration_bins":cal,"predictions":preds}
