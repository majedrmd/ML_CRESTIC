# -*- coding: utf-8 -*-
"""
CSV Sender — Encrypts and sends CSV to Observer
Runs on YOUR machine alongside Factory IO and secure_server_xyz.py

How it works:
  1. Waits for the CSV file to be ready (tutor stopped the machine)
  2. Encrypts the CSV using AES-256 (same technique as model weights)
  3. Sends the encrypted CSV to the observer machine
  4. Observer receives, decrypts, and stores it

Usage:
  Run this in a separate Spyder console alongside secure_server_xyz.py
  It waits automatically until you tell it to send
"""

from cryptography.fernet import Fernet
import socket
import os
import glob
import time

# ── CONFIG ────────────────────────────────────────────────────
# TODO: Replace with observer's real IP when available
OBSERVER_IP   = "10.145.23.74"   # tutor's machine (observer runs there)
OBSERVER_PORT = 5025              # port observer listens on
KEY_FILE      = "csv_secret.key"  # separate key just for CSV encryption

# ── KEY MANAGEMENT ────────────────────────────────────────────
def generate_csv_key():
    """Generate and save AES-256 key for CSV encryption."""
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"CSV encryption key generated and saved to '{KEY_FILE}'")
    print("Share this key with the observer securely (USB)!")
    return key

def load_csv_key():
    """Load existing key or generate a new one."""
    if not os.path.exists(KEY_FILE):
        print(f"Key file not found — generating new key...")
        return generate_csv_key()
    with open(KEY_FILE, "rb") as f:
        return f.read()

# ── FIND LATEST CSV ───────────────────────────────────────────
def find_latest_csv():
    """Find the most recently created pickplace CSV file."""
    csv_files = glob.glob("pickplace_records_*.csv")
    if not csv_files:
        return None
    # Sort by modification time — most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]

# ── ENCRYPT CSV ───────────────────────────────────────────────
def encrypt_csv(csv_path):
    """Encrypt CSV file and return encrypted bytes."""
    key    = load_csv_key()
    fernet = Fernet(key)

    with open(csv_path, "rb") as f:
        csv_bytes = f.read()

    encrypted = fernet.encrypt(csv_bytes)
    print(f"CSV encrypted: {len(csv_bytes)} bytes -> {len(encrypted)} bytes")
    return encrypted

# ── SEND TO OBSERVER ──────────────────────────────────────────
def send_to_observer(encrypted_data, csv_filename):
    """Send encrypted CSV to observer machine."""
    print(f"\nConnecting to observer at {OBSERVER_IP}:{OBSERVER_PORT}...")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((OBSERVER_IP, OBSERVER_PORT))

        # Send filename first (padded to 256 bytes)
        filename_bytes = csv_filename.encode().ljust(256)[:256]
        s.send(filename_bytes)

        # Send size of encrypted data (8 bytes)
        data_size = len(encrypted_data)
        s.send(data_size.to_bytes(8, byteorder='big'))

        # Send encrypted data
        s.sendall(encrypted_data)
        s.close()

        print(f"Encrypted CSV sent successfully to observer!")
        print(f"  File     : {csv_filename}")
        print(f"  Size     : {data_size} bytes (encrypted)")

    except ConnectionRefusedError:
        print(f"ERROR: Observer not listening on {OBSERVER_IP}:{OBSERVER_PORT}")
        print("Make sure observer.py is running on the observer machine!")
    except Exception as e:
        print(f"ERROR sending CSV: {e}")

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   CSV Sender — Encrypted CSV Transmission")
    print("=" * 55)
    print(f"   Observer  : {OBSERVER_IP}:{OBSERVER_PORT}")
    print(f"   Key file  : {KEY_FILE}")
    print("=" * 55)
    print()

    # Generate key if needed
    load_csv_key()
    print()

    while True:
        print("Options:")
        print("  1 - Find and send latest CSV to observer")
        print("  2 - Send specific CSV file")
        print("  3 - Exit")
        print()
        choice = input("Choose (1/2/3): ").strip()

        if choice == "1":
            csv_path = find_latest_csv()
            if csv_path is None:
                print("No CSV files found in current folder!")
                print("Make sure the XYZ scene ran and generated a CSV.")
            else:
                print(f"\nLatest CSV found: {csv_path}")
                confirm = input("Send this file? (y/n): ").strip().lower()
                if confirm == "y":
                    encrypted = encrypt_csv(csv_path)
                    send_to_observer(encrypted, os.path.basename(csv_path))

        elif choice == "2":
            csv_path = input("Enter CSV filename: ").strip()
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}")
            else:
                encrypted = encrypt_csv(csv_path)
                send_to_observer(encrypted, os.path.basename(csv_path))

        elif choice == "3":
            print("Exiting.")
            break

        print()
