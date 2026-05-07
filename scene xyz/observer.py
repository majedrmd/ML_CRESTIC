# -*- coding: utf-8 -*-
"""
Observer — Receives and decrypts CSV from Factory IO machine
Runs on TUTOR's machine (Console 2) or any third machine

How it works:
  1. Listens on port 5025 for incoming connections
  2. Receives encrypted CSV from student's machine
  3. Decrypts using the shared key (csv_secret.key)
  4. Saves the decrypted CSV locally
  5. Ready for LSTM anomaly detection (future step)

Files needed:
  - observer.py     (this file)
  - csv_secret.key  (student sends this separately — USB!)

Usage:
  Run this BEFORE csv_sender.py sends the file
  It waits automatically for incoming connections
"""

from cryptography.fernet import Fernet
import socket
import os
import datetime

# ── CONFIG ────────────────────────────────────────────────────
LISTEN_PORT  = 5025             # port to listen on
KEY_FILE     = "csv_secret.key" # same key as student's
SAVE_FOLDER  = "received_csv"   # folder to save received files

# ── KEY MANAGEMENT ────────────────────────────────────────────
def load_key():
    """Load the decryption key from file."""
    if not os.path.exists(KEY_FILE):
        print(f"ERROR: Key file '{KEY_FILE}' not found!")
        print("Ask the student to send you csv_secret.key (via USB).")
        import sys; sys.exit()
    with open(KEY_FILE, "rb") as f:
        return f.read()

# ── DECRYPT CSV ───────────────────────────────────────────────
def decrypt_csv(encrypted_data):
    """Decrypt received CSV bytes."""
    key    = load_key()
    fernet = Fernet(key)
    try:
        decrypted = fernet.decrypt(encrypted_data)
        return decrypted
    except Exception as e:
        print(f"DECRYPTION FAILED: {e}")
        print("Wrong key or corrupted data!")
        return None

# ── SAVE CSV ──────────────────────────────────────────────────
def save_csv(csv_bytes, filename):
    """Save decrypted CSV to local folder."""
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Add timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"received_{timestamp}_{filename}"
    save_path = os.path.join(SAVE_FOLDER, save_name)

    with open(save_path, "wb") as f:
        f.write(csv_bytes)

    return save_path

# ── HANDLE INCOMING CONNECTION ─────────────────────────────────
def handle_connection(conn, addr):
    print(f"\n>>> Incoming connection from {addr[0]}:{addr[1]}")

    try:
        # Receive filename (256 bytes)
        filename_bytes = conn.recv(256)
        filename       = filename_bytes.decode().strip()
        print(f"  Receiving file : {filename}")

        # Receive data size (8 bytes)
        size_bytes = conn.recv(8)
        data_size  = int.from_bytes(size_bytes, byteorder='big')
        print(f"  Expected size  : {data_size} bytes")

        # Receive encrypted data
        encrypted_data = b""
        while len(encrypted_data) < data_size:
            chunk = conn.recv(4096)
            if not chunk:
                break
            encrypted_data += chunk

        print(f"  Received       : {len(encrypted_data)} bytes")

        # Decrypt
        print(f"  Decrypting...")
        csv_bytes = decrypt_csv(encrypted_data)

        if csv_bytes is None:
            print("  ERROR: Decryption failed!")
            return

        # Save
        save_path = save_csv(csv_bytes, filename)
        print(f"  Saved to       : {save_path}")

        # Preview first few lines
        lines = csv_bytes.decode().split("\n")
        print(f"\n  CSV Preview (first 3 rows):")
        for line in lines[:4]:
            if line.strip():
                print(f"    {line}")

        print(f"\n  Total rows     : {len(lines)-1}")
        print(f"\n  File ready for LSTM anomaly detection!")

    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        conn.close()

# ── MAIN SERVER ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Observer — Encrypted CSV Receiver")
    print("=" * 55)
    print(f"   Listening on port : {LISTEN_PORT}")
    print(f"   Key file          : {KEY_FILE}")
    print(f"   Save folder       : {SAVE_FOLDER}/")
    print("=" * 55)
    print()

    # Verify key exists
    load_key()
    print("   Key loaded successfully!")
    print("   Waiting for incoming CSV from student's machine...")
    print()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", LISTEN_PORT))
    server.listen(5)

    while True:
        try:
            conn, addr = server.accept()
            handle_connection(conn, addr)
            print()
            print("Waiting for next CSV...")
        except KeyboardInterrupt:
            print("\nObserver stopped.")
            break
