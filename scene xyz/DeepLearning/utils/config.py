# -*- coding: utf-8 -*-
"""
config.py — Centralized configuration and model builder.
Used by train_lstm.py and diagnosis.py (and any future scripts).
To change hyperparameters: edit get_config() only.
"""

from utils.model_def import PickPlaceLSTM

def get_config():
    """
    Returns a dictionary with all hyperparameters and paths.
    Edit this function to adapt the model to a new scene or dataset.
    """
    return {
        "csv_path"     : "pickplace_records_1776331338.csv",
        "features"     : [
            "box_at_place", "part_at_place", "detected", "c_limit",
            "box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap"
        ],
        "n_past"       : 10,     # number of past states used as input
        "hidden_size"  : 64,     # LSTM hidden neurons
        "n_layers"     : 1,      # LSTM stacked layers
        "epochs"       : 100,    # max training epochs
        "lr"           : 0.001,  # learning rate
        "batch_size"   : 32,     # batch size
        "patience"     : 10,     # early stopping patience
        "weights_path" : "pickplace_lstm_weights.pth",
        "plot_path"    : "lstm_training_curves.png",
    }


def build_model(cfg):
    """
    Instantiate and return a PickPlaceLSTM model from a config dict.

    Args:
        cfg : dict returned by get_config()

    Returns:
        model : PickPlaceLSTM instance
    """
    return PickPlaceLSTM(
        input_size  = len(cfg["features"]),
        hidden_size = cfg["hidden_size"],
        n_layers    = cfg["n_layers"],
        output_size = len(cfg["features"])
    )
