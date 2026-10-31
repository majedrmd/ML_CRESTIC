# -*- coding: utf-8 -*-
"""
Train MLP on Pick and Place XYZ CSV data
Then encrypt the trained weights using AES-128 (Fernet)

Steps:
  1. Load CSV data
  2. Prepare inputs (sensors) and outputs (actuators)
  3. Train the MLP
  4. Evaluate the model
  5. Encrypt the trained weights
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from cryptography.fernet import Fernet
import pickle
import os

# ── CONFIG ────────────────────────────────────────────────────
CSV_PATH       = "../data/raw/pickplace_records_1776331338.csv"
ENCRYPTED_PATH = "pickplace_mlp_encrypted.bin"
KEY_PATH       = "pickplace_secret.key"
EPOCHS         = 100
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
TEST_SIZE      = 0.2   # 20% of data for testing

# ── STEP 1: LOAD DATA ─────────────────────────────────────────
print("=" * 55)
print("   MLP Training — Pick and Place XYZ")
print("=" * 55)
print()
print("Step 1 — Loading CSV data...")

df = pd.read_csv(CSV_PATH)
print(f"  Total rows loaded : {len(df)}")
print(f"  Columns           : {list(df.columns)}")
print()

# ── STEP 2: PREPARE INPUTS AND OUTPUTS ────────────────────────
print("Step 2 — Preparing inputs and outputs...")

# INPUTS — sensor columns (what we READ from the machine)
input_cols  = ["box_at_place", "part_at_place", "detected", "c_limit"]

# OUTPUTS — actuator columns (what we WRITE to the machine)
output_cols = ["box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap"]

X = df[input_cols].values.astype(np.float32)
Y = df[output_cols].values.astype(np.float32)

print(f"  Input  shape : {X.shape}  (rows x 4 sensors)")
print(f"  Output shape : {Y.shape}  (rows x 5 actuators)")
print()

# ── STEP 3: SPLIT INTO TRAIN AND TEST ─────────────────────────
print("Step 3 — Splitting data into train/test sets...")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=42)

print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")
print()

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── STEP 4: DEFINE MLP ARCHITECTURE ───────────────────────────
print("Step 4 — Building MLP architecture...")

class PickPlaceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 8),    # 4 inputs  -> 8 hidden neurons
            nn.ReLU(),          # ReLU activation on hidden layer
            nn.Linear(8, 5),    # 8 hidden  -> 5 outputs
            nn.Sigmoid()        # Sigmoid for binary outputs (0 or 1)
        )

    def forward(self, x):
        return self.network(x)

model = PickPlaceMLP()
print(f"  Architecture:")
print(f"    Input  layer : 4 neurons  (box_at_place, part_at_place, detected, c_limit)")
print(f"    Hidden layer : 8 neurons  (ReLU activation)")
print(f"    Output layer : 5 neurons  (box_conveyor, part_conveyor, exit_conveyor, c_plus, grap)")
print(f"    Output activation : Sigmoid (values between 0 and 1)")
print()

# ── STEP 5: TRAIN THE MODEL ───────────────────────────────────
print("Step 5 — Training the MLP...")
print(f"  Epochs        : {EPOCHS}")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Loss function : Binary Cross-Entropy (BCELoss)")
print(f"  Optimizer     : Adam")
print()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss        = criterion(predictions, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

print()

# ── STEP 6: EVALUATE THE MODEL ────────────────────────────────
print("Step 6 — Evaluating model on test data...")

model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    # Convert probabilities to binary (threshold 0.5)
    predicted_binary = (predictions >= 0.5).float()
    true_binary      = Y_test_t

# Calculate accuracy for each output
print("  Accuracy per actuator:")
actuator_names = output_cols
for i, name in enumerate(actuator_names):
    acc = accuracy_score(
        true_binary[:, i].numpy(),
        predicted_binary[:, i].numpy())
    print(f"    {name:<20} : {acc*100:.1f}%")

# Overall accuracy
overall = accuracy_score(
    true_binary.numpy().flatten(),
    predicted_binary.numpy().flatten())
print(f"\n  Overall accuracy : {overall*100:.1f}%")
print()

# ── STEP 7: ENCRYPT THE TRAINED WEIGHTS ──────────────────────
print("Step 7 — Encrypting trained model weights...")

# Generate encryption key
key    = Fernet.generate_key()
fernet = Fernet(key)

# Save the key
with open(KEY_PATH, "wb") as f:
    f.write(key)
print(f"  Key saved to '{KEY_PATH}' — keep this SECRET!")

# Serialize and encrypt weights
weights_bytes     = pickle.dumps(model.state_dict())
encrypted_weights = fernet.encrypt(weights_bytes)

# Save encrypted weights
with open(ENCRYPTED_PATH, "wb") as f:
    f.write(encrypted_weights)

print(f"  Original size  : {len(weights_bytes)} bytes")
print(f"  Encrypted size : {len(encrypted_weights)} bytes")
print(f"  Saved to '{ENCRYPTED_PATH}'")
print()

# ── STEP 8: VERIFY DECRYPTION WORKS ──────────────────────────
print("Step 8 — Verifying decryption...")

with open(ENCRYPTED_PATH, "rb") as f:
    enc = f.read()

decrypted_weights = pickle.loads(fernet.decrypt(enc))
model2 = PickPlaceMLP()
model2.load_state_dict(decrypted_weights)

# Check weights are identical
all_match = all(
    torch.equal(v1, v2)
    for (_, v1), (_, v2) in zip(
        model.state_dict().items(),
        model2.state_dict().items()))

print(f"  Weights match after decryption : {all_match}")
print()

# ── DONE ─────────────────────────────────────────────────────
print("=" * 55)
print("   Training and Encryption Complete!")
print("=" * 55)
print()
print("Files created:")
print(f"  {KEY_PATH:<35} <- SECRET key (never share publicly!)")
print(f"  {ENCRYPTED_PATH:<35} <- Encrypted model (safe to share)")
print()
print("Next step:")
print("  Use these files in secure_server.py to load")
print("  the model securely before running Factory IO.")
