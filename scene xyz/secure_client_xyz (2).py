# -*- coding: utf-8 -*-
"""
Secure Client — XYZ Pick and Place Control
Runs on TUTOR's machine

Connects to student's secure proxy on port 5020 (NOT 502 directly).
The proxy checks tutor's IP then forwards to Factory IO.

Files needed on tutor's machine:
  - secure_client_xyz.py  (this file)
  - pyModbusTCP installed (pip install pyModbusTCP)

Usage:
  1. Make sure secure_server_xyz.py is running on student's machine
  2. Make sure Factory IO XYZ scene is running on student's machine
  3. Run this file in Spyder
  4. Press emergency stop button to activate
  5. Press Start button to run the scene
"""

from pyModbusTCP.client import ModbusClient
import time
import csv

# ── CONNECTION CONFIG ─────────────────────────────────────────
STUDENT_IP = "172.20.10.2"  # student's machine IP
PROXY_PORT = 5020             # secure proxy port — NOT 502!

# ── PIN NUMBERS ───────────────────────────────────────────────
pin_box_at_place   = 0
pin_part_at_place  = 1
pin_detected       = 2
pin_c_limit        = 3
pin_start          = 4
pin_reset          = 5
pin_stop           = 6
pin_emergency_stop = 7

pin_read_x = 0
pin_read_y = 1
pin_read_z = 2

pin_box_conveyor  = 10
pin_part_conveyor = 11
pin_exit_conveyor = 12
pin_c_plus        = 13
pin_grap          = 14
pin_exit_green    = 15
pin_exit_yellow   = 16
pin_exit_red      = 17
pin_start_light   = 18
pin_stop_light    = 19

pin_spx = 0
pin_spy = 1
pin_spz = 2

# ── EVENT FUNCTION ────────────────────────────────────────────
def event(sensors):
    previous_state = int(sensors[0])
    state          = int(sensors[1])
    if state - previous_state == 0:    return None
    elif state - previous_state == 1:  return "RE"
    elif state - previous_state == -1: return "FE"
    else: raise ValueError("Error with the values")

# ── CSV SETUP ─────────────────────────────────────────────────
components = ["box_at_place", "part_at_place", "detected", "c_limit",
              "start", "reset", "stop", "emergency_stop",
              "box_conveyor", "part_conveyor", "exit_conveyor", "c_plus", "grap"]
past_observation = None
end_name  = int(time.time())
filename  = f'pickplace_records_{end_name}.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(components + ["time"])

# ── CONNECT THROUGH SECURE PROXY ─────────────────────────────
print("=" * 55)
print("   SECURE CLIENT — XYZ Pick and Place")
print("=" * 55)
print(f"   Connecting to {STUDENT_IP}:{PROXY_PORT}...")
print(f"   (through secure proxy — not direct Modbus)")
print()

c = ModbusClient(STUDENT_IP, port=PROXY_PORT, auto_open=True, auto_close=False)
c.open()

if not c.is_open:
    print("ERROR: Could not connect to secure proxy!")
    print("Make sure secure_server_xyz.py is running on student's machine.")
    import sys
    sys.exit()

print("   Connected through secure proxy!")
print("=" * 55)
print()

# ── SAFE READ HELPER ─────────────────────────────────────────
def safe_read_di(pin, default=False):
    """Read discrete input safely — returns default if None."""
    result = c.read_discrete_inputs(pin)
    return result[0] if result else default

def safe_read_ir(pin, default=0):
    """Read input register safely — returns default if None."""
    result = c.read_input_registers(pin)
    return result[0] if result else default

# ── MAIN CONTROL LOOP ─────────────────────────────────────────
if c.is_open:

    # Initialization
    box_at_place   = [safe_read_di(pin_box_at_place)] * 2
    part_at_place  = [safe_read_di(pin_part_at_place)] * 2
    detected       = [safe_read_di(pin_detected)] * 2
    c_limit        = [safe_read_di(pin_c_limit)] * 2
    start          = [safe_read_di(pin_start)] * 2
    reset          = [safe_read_di(pin_reset)] * 2
    stop           = [safe_read_di(pin_stop)] * 2
    emergency_stop = safe_read_di(pin_emergency_stop)
    x = [safe_read_ir(pin_read_x)] * 2
    y = [safe_read_ir(pin_read_y)] * 2
    z = [safe_read_ir(pin_read_z)] * 2

    # Reset outputs
    box_conveyor  = False
    part_conveyor = False
    grap          = False
    spx = spy = spz = 0
    c_plus        = False
    exit_conveyor = False
    c.write_single_coil(pin_box_conveyor,  box_conveyor)
    c.write_single_coil(pin_part_conveyor, part_conveyor)
    c.write_single_coil(pin_grap,          grap)
    c.write_single_coil(pin_c_plus,        c_plus)
    c.write_single_register(pin_spx, spx)
    c.write_single_register(pin_spy, spy)
    c.write_single_register(pin_spz, spz)
    c.write_single_coil(pin_exit_conveyor, exit_conveyor)

    task1 = False
    task2 = False
    task3 = False
    task4 = False
    task6 = False
    task5 = True
    mode  = "off"
    init  = True

    print(f"Ready — emergency_stop={emergency_stop}")

    while c.is_open:

        # Read sensors
        box_at_place   = [box_at_place[-1],  safe_read_di(pin_box_at_place)]
        part_at_place  = [part_at_place[-1], safe_read_di(pin_part_at_place)]
        detected       = [detected[-1],      safe_read_di(pin_detected)]
        c_limit        = [c_limit[-1],       safe_read_di(pin_c_limit)]
        start          = [start[-1],         safe_read_di(pin_start)]
        reset          = [reset[-1],         safe_read_di(pin_reset)]
        stop           = [stop[-1],          safe_read_di(pin_stop)]
        emergency_stop = safe_read_di(pin_emergency_stop)
        x = [x[-1], safe_read_ir(pin_read_x)]
        y = [y[-1], safe_read_ir(pin_read_y)]
        z = [z[-1], safe_read_ir(pin_read_z)]

        if event(start) == "RE": mode = "on"
        if event(stop)  == "RE": mode = "off"

        if mode == "on":

            # T5: Bring box
            if task5 == True:
                if box_at_place[-1] == True:
                    box_conveyor = True
                elif event(box_at_place) == "FE":
                    box_conveyor  = False
                    exit_conveyor = False
                    task1 = True
                    task5 = False

            # T2: Bring part
            if task3 == True or init == True:
                if part_at_place[-1] == True:
                    task2 = 10
                    part_conveyor = True
                elif event(part_at_place) == "FE":
                    part_conveyor = False
                    task2 = True
                    init  = False
                elif task2 == True and task4 == True:
                    task3 = False

            # T3: Pick part
            if task6 == True and task2 == True:
                if z[-1] != 5 and grap == False:
                    spz   = 6
                    task3 = 10
                elif z[-1] == 5 and grap == False and task3 == 10:
                    grap  = True
                    task3 = 20
                elif grap == True and z[-1] != 0 and task3 == 20:
                    spz   = 0
                    task3 = 30
                elif z[-1] == 0 and task3 == 30:
                    task3 = True
                    task6 = False
                    task2 = False

            # T4: Place 3 parts
            if task1 == True and task3 == True:
                if x[-1] != 3 and task4 == False:
                    spx   = 3
                    task4 = 10
                elif x[-1] == 3 and z[-1] != 10 and task4 == 10:
                    spy   = 6
                    spz   = 10
                    task4 = 20
                elif y[-1] == spy-1 and z[-1] == 10 and grap == True and task4 == 20:
                    grap  = False
                    task4 = 30
                elif grap == False and z[-1] != 0 and task4 == 30:
                    spz   = 0
                    task4 = 40
                elif task4 == 40 and z[-1] == 0:
                    task3 = False
                    task4 = 50

                elif task4 == 50 and x[-1] != 3:
                    spx   = 3
                    task4 = 60
                elif x[-1] == 3 and y[-1] != 3 and task4 == 60:
                    spy   = 3
                    task4 = 70
                elif y[-1] == 3 and z[-1] != 10 and task4 == 70:
                    spz   = 10
                    task4 = 75
                elif y[-1] == spy and z[-1] == 10 and grap == True and task4 == 75:
                    grap  = False
                    task4 = 80
                elif grap == False and z[-1] != 0 and task4 == 80:
                    spz   = 0
                    task4 = 90
                elif task4 == 90 and z[-1] == 0:
                    task3 = False
                    task4 = 100

                elif task4 == 100 and x[-1] != 3:
                    c_plus = True
                    spx    = 3
                    task4  = 110
                elif x[-1] == 3 and z[-1] != 5 and task4 == 110:
                    spz   = 5
                    task4 = 120
                elif z[-1] == 5 and grap == True and task4 == 120:
                    grap  = False
                    task4 = 130
                elif grap == False and z[-1] != 0 and task4 == 130:
                    c_plus = False
                    spz    = 0
                    task4  = 140
                elif task4 == 140 and z[-1] == 0:
                    task3 = False
                    task4 = True

            # T5: Evacuate box
            if task4 == True:
                if box_at_place[-1] == False:
                    exit_conveyor = True
                    box_conveyor  = True
                elif event(box_at_place) == "RE":
                    task5 = True
                    task4 = False

            # T6: Go to pick position
            if (task4 == 50 or task4 == 100 or task4 == True or init) and task3 == False:
                if x[-1] != 7 or (y[-1] != 4 and y[-1] != 5) or z[-1] != 0:
                    spx   = 8
                    spy   = 5
                    spz   = 0
                    task6 = 10
                elif x[-1] == 7 and (y[-1] == 4 or y[-1] == 5) and z[-1] == 0:
                    task6 = True
                elif task5 == True and task6 == True:
                    task4 = False

        elif mode == "off":
            box_conveyor  = False
            part_conveyor = False
            exit_conveyor = False

        # Write outputs
        c.write_single_coil(pin_box_conveyor,  box_conveyor)
        c.write_single_coil(pin_part_conveyor, part_conveyor)
        c.write_single_coil(pin_grap,          grap)
        c.write_single_coil(pin_c_plus,        c_plus)
        c.write_single_register(pin_spx, spx)
        c.write_single_register(pin_spy, spy)
        c.write_single_register(pin_spz, spz)
        c.write_single_coil(pin_exit_conveyor, exit_conveyor)

        # Save CSV data
        observation = [box_at_place[-1], part_at_place[-1], detected[-1],
                       c_limit[-1], start[-1], reset[-1], stop[-1],
                       emergency_stop, box_conveyor, part_conveyor,
                       exit_conveyor, c_plus, grap]
        observation = list(map(lambda x: 0 if x==False else 1 if x==True else x, observation))
        if observation != past_observation:
            past_observation = observation.copy()
            observation.append(time.time())
            with open(filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(observation)
