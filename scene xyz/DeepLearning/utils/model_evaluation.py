import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report)

import matplotlib.pyplot as plt

def global_evaluate(Y_true, Y_pred,):
    """global_evaluate
    Args:
        Y_true (array or tensor): true labels
        Y_pred (array or tensor): predicted labels

    Returns:
        dict: evaluation metrics dictionary
    """

    # ── GLOBAL METRICS ────────────────────────────────────────────
    # Classical accuracy (flat) — each of x features independently
    classical_acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())

    # Exact Match Accuracy — whole vector must be correct simultaneously
    exact_match = np.all(Y_true == Y_pred, axis=1).mean()

    # Global metrics (macro average across all features)fn and fp

    global_prec = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
    global_rec  = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
    global_f1   = f1_score(Y_true, Y_pred, average='macro', zero_division=0)

    # Micro average (global across all predictions)
    micro_prec = precision_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    micro_rec  = recall_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)
    micro_f1   = f1_score(Y_true.flatten(), Y_pred.flatten(), zero_division=0)

    return {
        'classical_accuracy': classical_acc,
        'exact_match': exact_match,
        'global_precision': global_prec,
        'global_recall': global_rec,
        'global_f1': global_f1,
        'micro_precision': micro_prec,
        'micro_recall': micro_rec,
        'micro_f1': micro_f1
    }



def component_evaluate(Y_true, Y_pred, FEATURES = None):

    if not FEATURES:
        FEATURES = [f"Feature {i}" for i in range(Y_true.shape[1])]

    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    for i, name in enumerate(FEATURES):
        y_true_i = Y_true[:, i]
        y_pred_i = Y_pred[:, i]

        acc  = accuracy_score(y_true_i, y_pred_i)
        prec = precision_score(y_true_i, y_pred_i, zero_division=0)
        rec  = recall_score(y_true_i, y_pred_i, zero_division=0)
        f1   = f1_score(y_true_i, y_pred_i, zero_division=0)

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

    return {
        'feature_names': FEATURES, 
        'accuracy': all_acc,
        'precision': all_prec,  
        'recall': all_rec,
        'f1': all_f1
    }


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, path:str = None,
                          N_PAST:int=None, HIDDEN_SIZE=None, PATIENCE=None, system_name=None):
    # ── STEP 6: PLOT ──────────────────────────────────────────────
    print("Step 6 — Plotting loss and accuracy curves...")
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs_range, val_losses,   'r-', linewidth=2, label='Validation Loss')

    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (BCE)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs_range, val_accs,   'r-', linewidth=2, label='Validation Accuracy')

    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.suptitle(f'LSTM Training — {system_name}\n'
                f'N_PAST={N_PAST} | Hidden={HIDDEN_SIZE} | Patience={PATIENCE}',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    if not path: path = "../figures/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    