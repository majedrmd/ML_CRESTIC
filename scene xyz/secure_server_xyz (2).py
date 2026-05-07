# -*- coding: utf-8 -*-
"""
Secure Server — XYZ Pick and Place
Runs on YOUR machine alongside Factory IO

Security Layer 1: IP Whitelist
  - Listens on port 5020
  - Checks tutor's IP against whitelist
  - If authorized  -> forwards to Factory IO port 502
  - If unauthorized -> refuses + logs alert

CSV Recording:
  - Records all Modbus traffic passing through the proxy
  - Saves sensor + actuator states to CSV automatically
  - CSV is ready to send to observer after session ends

Usage:
  1. Open Factory IO, load XYZ scene
  2. Driver -> Modbus TCP/IP Server -> Connect -> Play
  3. Run this file in Spyder
  4. Leave it running — tutor connects from his machine
  5. After tutor stops -> CSV is complete -> run csv_sender.py
"""

import socket
import threading
import datetime
import struct
import csv
import time
import os

# ── SECURITY CONFIG ───────────────────────────────────────────
TUTOR_IP      = "10.145.23.74"   # tutor's real IP

ALLOWED_IPS   = [
    TUTOR_IP,
    "127.0.0.1",       # localhost — for local testing
    "10.129.20.16", 
    "172.20.10.2"# your own IP — for local testing
]

PROXY_PORT       = 5020
FACTORY_IO_IP = "172.20.10.2"
FACTORY_IO_PORT  = 502

# ── CSV CONFIG ────────────────────────────────────────────────
CSV_COLUMNS = [
    "box_at_place", "part_at_place", "detected", "c_limit",
    "start", "reset", "stop", "emergency_stop",
    "box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap",
    "time"
]

# Pin mapping — sensors (discrete inputs) and actuators (coils)
SENSOR_PINS   = {0: "box_at_place", 1: "part_at_place", 2: "detected",
                  3: "c_limit", 4: "start", 5: "reset", 6: "stop", 7: "emergency_stop"}
ACTUATOR_PINS = {10: "box_conveyor", 11: "part_conveyor", 12: "exit_conveyor",
                  13: "c_plus", 14: "grap"}

# ── SHARED STATE ──────────────────────────────────────────────
machine_state    = {col: 0 for col in CSV_COLUMNS if col != "time"}
past_observation = None
csv_lock         = threading.Lock()
csv_filename     = None
csv_writer_obj   = None
csv_file_obj     = None

def init_csv():
    global csv_filename, csv_writer_obj, csv_file_obj
    timestamp    = int(time.time())
    csv_filename = f"pickplace_records_{timestamp}.csv"
    csv_file_obj    = open(csv_filename, 'w', newline='')
    csv_writer_obj  = csv.writer(csv_file_obj)
    csv_writer_obj.writerow(CSV_COLUMNS)
    print(f"CSV recording started: {csv_filename}")

def save_state():
    global past_observation
    with csv_lock:
        observation = [machine_state[col] for col in CSV_COLUMNS if col != "time"]
        if observation != past_observation:
            past_observation = observation.copy()
            row = observation + [time.time()]
            csv_writer_obj.writerow(row)
            csv_file_obj.flush()

def close_csv():
    if csv_file_obj:
        csv_file_obj.close()
        print(f"CSV recording complete: {csv_filename}")

# ── MODBUS PACKET PARSER ──────────────────────────────────────
def parse_modbus(data, direction):
    """
    Parse Modbus TCP packets to extract sensor and actuator states.
    direction: "REQ" = tutor->factory (write commands)
               "RES" = factory->tutor (read responses)
    """
    try:
        if len(data) < 8:
            return

        # Modbus TCP header: transaction_id(2) + protocol(2) + length(2) + unit_id(1) + func_code(1)
        func_code = data[7]

        # Function code 1 = Read Coils (actuators)
        # Function code 2 = Read Discrete Inputs (sensors)
        # Function code 5 = Write Single Coil (actuator command)
        # Function code 15 = Write Multiple Coils

        if func_code == 5 and direction == "REQ" and len(data) >= 12:
            # Write single coil — tutor sending actuator command
            coil_addr  = struct.unpack(">H", data[8:10])[0]
            coil_value = struct.unpack(">H", data[10:12])[0]
            value      = 1 if coil_value == 0xFF00 else 0
            if coil_addr in ACTUATOR_PINS:
                col = ACTUATOR_PINS[coil_addr]
                with csv_lock:
                    machine_state[col] = value
                save_state()

        elif func_code == 2 and direction == "RES" and len(data) >= 10:
            # Read discrete inputs response — sensor values
            byte_count = data[8]
            if byte_count >= 1 and len(data) >= 9 + byte_count:
                bits = data[9]
                for pin, name in SENSOR_PINS.items():
                    if pin < 8:
                        value = (bits >> pin) & 1
                        with csv_lock:
                            machine_state[name] = value
                save_state()

    except Exception:
        pass

# ── LOGGING ───────────────────────────────────────────────────
def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    with open("security_log_xyz.txt", "a") as f:
        f.write(line + "\n")

def log_alert(ip, reason):
    msg = f"SECURITY ALERT | IP: {ip} | {reason}"
    log(msg, "ALERT")
    with open("alerts_xyz.txt", "a") as f:
        f.write(msg + "\n")

# ── LAYER 1: IP WHITELIST ─────────────────────────────────────
def check_ip(client_ip):
    if client_ip in ALLOWED_IPS:
        log(f"ACCESS GRANTED to {client_ip}", "OK")
        return True
    else:
        log_alert(client_ip, "IP not in whitelist — connection refused")
        return False

# ── PROXY HANDLER ─────────────────────────────────────────────
def handle_client(client_socket, client_address):
    client_ip = client_address[0]
    log(f"Connection attempt from {client_ip}:{client_address[1]}")

    if not check_ip(client_ip):
        print(f"\n{'='*55}")
        print(f"  UNAUTHORIZED: {client_ip} — CONNECTION REFUSED")
        print(f"{'='*55}\n")
        try:
            client_socket.send(b"UNAUTHORIZED")
        except:
            pass
        client_socket.close()
        return

    print(f"\n>>> Authorized connection from {client_ip}")
    print(f">>> Forwarding to Factory IO XYZ...")
    print(f">>> Recording CSV: {csv_filename}")

    try:
        factory_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        factory_socket.connect((FACTORY_IO_IP, FACTORY_IO_PORT))
        log(f"Connected to Factory IO at {FACTORY_IO_IP}:{FACTORY_IO_PORT}")

        def forward(src, dst, direction):
            try:
                src.settimeout(None)  # no timeout — keep alive
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    # Parse Modbus to record CSV
                    parse_modbus(data, direction)
                    dst.sendall(data)
            except Exception as e:
                pass
            finally:
                try: src.close()
                except: pass
                try: dst.close()
                except: pass

        t1 = threading.Thread(
            target=forward,
            args=(client_socket, factory_socket, "REQ"),
            daemon=True)
        t2 = threading.Thread(
            target=forward,
            args=(factory_socket, client_socket, "RES"),
            daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    except ConnectionRefusedError:
        print("ERROR: Cannot connect to Factory IO!")
        print("Make sure Factory IO is running with Modbus TCP/IP Server connected.")
    except Exception as e:
        log(f"Proxy error: {e}", "ERROR")
    finally:
        try: client_socket.close()
        except: pass
        log(f"Connection from {client_ip} closed")
        print(f"\nSession ended. CSV saved: {csv_filename}")
        print("You can now run csv_sender.py to send it to the observer!")

# ── MAIN SERVER ───────────────────────────────────────────────
def start_server():
    init_csv()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PROXY_PORT))
    server.listen(5)

    print("=" * 55)
    print("   SECURE MODBUS PROXY — XYZ Pick and Place")
    print("   Layer 1: IP Whitelist + CSV Recording")
    print("=" * 55)
    print(f"   Listening on      : port {PROXY_PORT}")
    print(f"   Factory IO        : {FACTORY_IO_IP}:{FACTORY_IO_PORT}")
    print(f"   Authorized IPs    :")
    for ip in ALLOWED_IPS:
        print(f"     - {ip}")
    print(f"   CSV file          : {csv_filename}")
    print("=" * 55)
    print("   Waiting for tutor connection...")
    print("   (all attempts logged to security_log_xyz.txt)")
    print()

    while True:
        try:
            client_socket, client_address = server.accept()
            thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address),
                daemon=True)
            thread.start()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            close_csv()
            break

if __name__ == "__main__":
    start_server()
