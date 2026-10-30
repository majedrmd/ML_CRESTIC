# -*- coding: utf-8 -*-
"""
diagnosis.py — Live anomaly detection on Factory I/O via Modbus.
Loads the trained LSTM weights and reads live sensor data.
At each cycle, it predicts the next state and compares it to the real one.
If they don't match -> anomaly flagged.

Requirements:
    - Factory I/O running with Modbus TCP/IP Server driver
    - Trained weights in models/pickplace_lstm_weights.pth
"""

import time
import torch
import numpy as np
from collections import deque
from pyModbusTCP.client import ModbusClient
from utils.config import get_config, build_model

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODBUS_IP = "10.129.20.16"
MODBUS_PORT    = 502
ANOMALY_THRESHOLD = 0.5     # prediction confidence threshold
POLL_INTERVAL  = 0.1        # seconds between each Modbus read

# Modbus coil pins (READ) — same as pick_place.py
PIN_BOX_AT_PLACE   = 0
PIN_PART_AT_PLACE  = 1
PIN_DETECTED       = 2
PIN_C_LIMIT        = 3
PIN_BOX_CONVEYOR   = 10
PIN_PART_CONVEYOR  = 11
PIN_EXIT_CONVEYOR  = 12
PIN_C_PLUS         = 13
PIN_GRAP           = 14

# Order must match FEATURES in config
COIL_PINS = [
    PIN_BOX_AT_PLACE,
    PIN_PART_AT_PLACE,
    PIN_DETECTED,
    PIN_C_LIMIT,
    PIN_BOX_CONVEYOR,
    PIN_PART_CONVEYOR,
    PIN_EXIT_CONVEYOR,
    PIN_C_PLUS,
    PIN_GRAP,
]


# ══════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════

def read_sensors(client, pins):
    """
    Read coil values from Modbus and return as a list of 0/1.

    Args:
        client : ModbusClient instance
        pins   : list of coil pin indices to read

    Returns:
        state : list of int (0 or 1), one per feature
                returns None if Modbus read fails
    """
    # Read enough coils to cover all pins in one call
    coils = client.read_coils(0, max(pins) + 1)
    if coils is None:
        return None
    return [int(coils[p]) for p in pins]


def predict_next_state(model, window):
    """
    Run the LSTM on the last N_PAST states and return predicted next state.

    Args:
        model  : trained PickPlaceLSTM
        window : deque of N_PAST states, each a list of n_features values

    Returns:
        prediction : numpy array of shape (n_features,) with values 0 or 1
    """
    x = torch.tensor(np.array(window), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
    return (output.numpy()[0] >= 0.5).astype(int)


def detect_anomaly(predicted, real, feature_names):
    """
    Compare predicted state to real state and flag mismatches.

    Args:
        predicted     : numpy array of predicted next state
        real          : list of real next state values
        feature_names : list of feature name strings

    Returns:
        anomalies : list of feature names that mismatch (empty if none)
    """
    anomalies = []
    for i, name in enumerate(feature_names):
        if predicted[i] != real[i]:
            anomalies.append(name)
    return anomalies


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load config and model
    cfg   = get_config()
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg["weights_path"]))
    model.eval()
    print("=" * 60)
    print("   LSTM Diagnosis — Pick and Place XYZ")
    print("=" * 60)
    print(f"  Weights loaded from : {cfg['weights_path']}")
    print(f"  Features monitored  : {cfg['features']}")
    print(f"  N_PAST              : {cfg['n_past']}")
    print()

    # 2. Connect to Factory I/O
    print(f"  Connecting to Factory I/O at {MODBUS_IP}:{MODBUS_PORT} ...")
    client = ModbusClient(MODBUS_IP, port=MODBUS_PORT)
    client.open()
    if not client.is_open:
        print("  ERROR: Cannot connect to Factory I/O.")
        print("  Make sure Factory I/O is running with Modbus TCP/IP Server driver.")
        raise SystemExit(1)
    print("  Connected! Starting live diagnosis...")
    print()
    print(f"  {'Cycle':<8} {'Status':<12} {'Anomalies'}")
    print(f"  {'-'*8} {'-'*12} {'-'*40}")

    # 3. Fill the initial window with N_PAST real states
    window = deque(maxlen=cfg["n_past"])
    print("  Filling initial window...")
    while len(window) < cfg["n_past"]:
        state = read_sensors(client, COIL_PINS)
        if state is not None:
            window.append(state)
        time.sleep(POLL_INTERVAL)
    print("  Window ready. Diagnosis running...\n")

    # 4. Main diagnosis loop
    cycle        = 0
    anomaly_log  = []

    try:
        while True:
            cycle += 1

            # Predict next state from current window
            predicted = predict_next_state(model, window)

            # Read real next state from Factory I/O
            time.sleep(POLL_INTERVAL)
            real_state = read_sensors(client, COIL_PINS)
            if real_state is None:
                print(f"  {cycle:<8} {'READ ERROR':<12}")
                continue

            # Detect anomalies
            anomalies = detect_anomaly(predicted, real_state, cfg["features"])

            # Log and print
            if anomalies:
                status = "⚠ ANOMALY"
                log_entry = {
                    "cycle"    : cycle,
                    "time"     : time.strftime("%H:%M:%S"),
                    "predicted": predicted.tolist(),
                    "real"     : real_state,
                    "anomalies": anomalies
                }
                anomaly_log.append(log_entry)
                print(f"  {cycle:<8} {status:<12} {anomalies}")
            else:
                status = "OK"
                print(f"  {cycle:<8} {status:<12} —")

            # Slide the window forward with the real state
            window.append(real_state)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("   Diagnosis stopped.")
        print("=" * 60)
        print(f"  Total cycles    : {cycle}")
        print(f"  Total anomalies : {len(anomaly_log)}")
        if anomaly_log:
            print("\n  Anomaly summary:")
            for entry in anomaly_log:
                print(f"    [{entry['time']}] Cycle {entry['cycle']} — {entry['anomalies']}")
        client.close()