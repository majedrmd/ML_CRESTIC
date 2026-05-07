# -*- coding: utf-8 -*-
"""
plot.py — Plot and save training loss and accuracy curves.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_curves(train_losses, val_losses, train_accs, val_accs,
                best_epoch, n_past, hidden_size, patience,
                save_path="lstm_training_curves.png"):
    """
    Plot loss and accuracy curves and save as image.

    Args:
        train_losses, val_losses  : loss history per epoch
        train_accs,   val_accs    : accuracy history per epoch
        best_epoch                : epoch of best model (for vertical line)
        n_past, hidden_size, patience : hyperparameters for title
        save_path                 : where to save the plot
    """
    epochs_range = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs_range, val_losses,   'r-', linewidth=2, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='green', linestyle='--',
                linewidth=2, label=f'Best model (epoch {best_epoch})')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (BCE)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs_range, val_accs,   'r-', linewidth=2, label='Validation Accuracy')
    ax2.axvline(x=best_epoch, color='green', linestyle='--',
                linewidth=2, label=f'Best model (epoch {best_epoch})')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.suptitle(
        f'LSTM Training — Pick and Place XYZ\n'
        f'N_PAST={n_past} | Hidden={hidden_size} | Patience={patience}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to '{save_path}'")
