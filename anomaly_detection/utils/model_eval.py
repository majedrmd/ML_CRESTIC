# -*- coding: utf-8 -*-
"""
model_eval.py — Evaluate the trained model on the test set.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)


def evaluate_model(model, X_test_t, Y_test_t, feature_names):
    """
    Evaluate model on test data.
    Computes per-feature and global metrics.

    Args:
        model         : trained PickPlaceLSTM
        X_test_t      : test input tensor
        Y_test_t      : test output tensor
        feature_names : list of feature names

    Returns:
        results dict with all metrics
    """
    import torch
    model.eval()
    with torch.no_grad():
        preds       = model(X_test_t)
        pred_binary = (preds >= 0.5).float().numpy()

    Y_true = Y_test_t.numpy()

    print(f"\n  {'Feature':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

    per_feature = {}
    for i, name in enumerate(feature_names):
        acc  = accuracy_score(Y_true[:, i], pred_binary[:, i])
        prec = precision_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        rec  = recall_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        f1   = f1_score(Y_true[:, i], pred_binary[:, i], zero_division=0)
        per_feature[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
        print(f"  {name:<25} {acc*100:>8.1f}% {prec*100:>9.1f}% "
              f"{rec*100:>7.1f}% {f1*100:>7.1f}%")

    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

    # Global metrics
    classical_acc = accuracy_score(Y_true.flatten(), pred_binary.flatten())
    exact_match   = np.all(Y_true == pred_binary, axis=1).mean()
    macro_prec    = precision_score(Y_true, pred_binary, average='macro', zero_division=0)
    macro_rec     = recall_score(Y_true, pred_binary, average='macro', zero_division=0)
    macro_f1      = f1_score(Y_true, pred_binary, average='macro', zero_division=0)
    micro_prec    = precision_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)
    micro_rec     = recall_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)
    micro_f1      = f1_score(Y_true.flatten(), pred_binary.flatten(), zero_division=0)

    print(f"\n  {'Classical Accuracy (flat)':<35} : {classical_acc*100:.2f}%")
    print(f"  {'Exact Match Accuracy (whole vector)':<35} : {exact_match*100:.2f}%")
    print(f"\n  {'Macro Precision':<35} : {macro_prec*100:.2f}%")
    print(f"  {'Macro Recall':<35}    : {macro_rec*100:.2f}%")
    print(f"  {'Macro F1-Score':<35}  : {macro_f1*100:.2f}%")
    print(f"\n  {'Micro Precision':<35} : {micro_prec*100:.2f}%")
    print(f"  {'Micro Recall':<35}    : {micro_rec*100:.2f}%")
    print(f"  {'Micro F1-Score':<35}  : {micro_f1*100:.2f}%")

    return dict(
        per_feature   = per_feature,
        classical_acc = classical_acc,
        exact_match   = exact_match,
        macro_prec    = macro_prec,
        macro_rec     = macro_rec,
        macro_f1      = macro_f1,
        micro_prec    = micro_prec,
        micro_rec     = micro_rec,
        micro_f1      = micro_f1,
    )
