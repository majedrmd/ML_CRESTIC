# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:14:58 2025

@author: waboa

FIXED: Pin addresses updated to match Factory I/O Modbus mapping (screenshot 2)
       IP updated to 10.129.20.16:502
"""

#Import libraries
from pyModbusTCP.client import ModbusClient
import time
import csv


class Component():
    def __init__(self,initial_state):
        self.state = initial_state
        
    def __call__(self, state):
        #Store the previous state
        previous_state = self.state
        self.state = state
        
        #Check the actual state
        if self.state - previous_state == 0: return None   # No event
        elif self.state - previous_state == 1 : return 10  # Rising edge
        elif self.state - previous_state == -1 : return -10 # Falling edge  
        else : raise ValueError("Error with the values")
        
        
def event(sensors):
    #Store the previous state
    previous_state = int(sensors[0])
    state = int(sensors[1])
    
    #Check the actual state
    if state - previous_state == 0: return None   # No event
    elif state - previous_state == 1 : return "RE"  # Rising edge
    elif state - previous_state == -1 : return "FE" # Falling edge  
    else : raise ValueError("Error with the values")
    
    
sensors = ["box_at_place", "part_at_place", "detected", "c_limit", "start",
              "reset", "stop", "emergency_stop"]
actuators = ["box_conveyor","part_conveyor", "exit_conveyor", "c_plus", "grap",
             "exit_green", "exit_yellow", "exit_red", "start_light", "stop_light"]

# ─────────────────────────────────────────────
# FIXED PIN ADDRESSES — matched to Factory I/O
# ─────────────────────────────────────────────

# SENSORS — Discrete Inputs (read with read_discrete_inputs)
pin_box_at_place   = 0   # Input 0
pin_part_at_place  = 1   # Input 1
pin_detected       = 2   # Input 2
pin_c_limit        = 3   # Input 3
pin_start          = 4   # Input 4
pin_reset          = 5   # Input 5
pin_stop           = 6   # Input 6
pin_emergency_stop = 7   # Input 7

# SENSORS — Position registers (read with read_input_registers)
pin_read_x = 0   # Input Reg 0
pin_read_y = 1   # Input Reg 1
pin_read_z = 2   # Input Reg 2

# ACTUATORS — Coils (write with write_single_coil)
pin_box_conveyor  = 10  # Coil 10
pin_part_conveyor = 11  # Coil 11
pin_exit_conveyor = 12  # Coil 12
pin_c_plus        = 13  # Coil 13
pin_grap          = 14  # Coil 14
pin_exit_green    = 15  # Coil 15
pin_exit_yellow   = 16  # Coil 16
pin_exit_red      = 17  # Coil 17
pin_start_light   = 18  # Coil 18
pin_stop_light    = 19  # Coil 19

# ACTUATORS — Setpoint registers (write with write_single_register)
pin_spx = 0   # Holding Reg 0
pin_spy = 1   # Holding Reg 1
pin_spz = 2   # Holding Reg 2

# ─────────────────────────────────────────────


components = ["box_at_place", "part_at_place","detected", "c_limit", "start",
               "reset", "stop", "emergency_stop", "box_conveyor", 
               "part_conveyor", "exit_conveyor", "c_plus", "grap"]    
past_observation = None
end_name = int(time.time())
filename = f'pickplace_records_{end_name}.csv'
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(components+["time"])

# FIXED IP ADDRESS — matched to Factory I/O driver config
c = ModbusClient("10.129.20.16", port=502)
c.open()
initialization = True

d = {False : 0, True : 1}
while c.open():
    
    """Initialization of sensors values"""
    box_at_place = [c.read_discrete_inputs(pin_box_at_place)[0]]*2
    part_at_place = [c.read_discrete_inputs(pin_part_at_place)[0], 
                     c.read_discrete_inputs(pin_part_at_place)[0]]
    detected = [c.read_discrete_inputs(pin_detected)[0], 
                c.read_discrete_inputs(pin_detected)[0]]
    c_limit = [c_limit[-1], c.read_discrete_inputs(pin_c_limit)[0]] if 'c_limit' in dir() else [c.read_discrete_inputs(pin_c_limit)[0]]*2
    c_limit = [c.read_discrete_inputs(pin_c_limit)[0], 
               c.read_discrete_inputs(pin_c_limit)[0]]
    start = [c.read_discrete_inputs(pin_start)[0], 
             c.read_discrete_inputs(pin_start)[0]]
    reset = [c.read_discrete_inputs(pin_reset)[0], 
             c.read_discrete_inputs(pin_reset)[0]]
    stop = [c.read_discrete_inputs(pin_stop)[0], 
            c.read_discrete_inputs(pin_stop)[0]]
    emergency_stop = c.read_discrete_inputs(pin_emergency_stop)[0]
    x = [c.read_input_registers(pin_read_x)[0]]*2
    y = [c.read_input_registers(pin_read_y)[0]]*2
    z = [c.read_input_registers(pin_read_z)[0]]*2


    """Writing the outputs"""
    box_conveyor = False
    part_conveyor = False
    grap = False
    spx = 0
    spy = 0
    spz = 0
    c_plus = False
    exit_conveyor = False
    c.write_single_coil(pin_box_conveyor, box_conveyor)
    c.write_single_coil(pin_part_conveyor, part_conveyor) 
    c.write_single_coil(pin_grap, grap)    
    c.write_single_coil(pin_c_plus, c_plus)    
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

    mode = "off"
    
    init = True
    
    while emergency_stop == True:
        
        """Reading the sensors"""
        box_at_place = [box_at_place[-1], c.read_discrete_inputs(pin_box_at_place)[0]]  # FIXED: was hardcoded 0
        part_at_place = [part_at_place[-1], c.read_discrete_inputs(pin_part_at_place)[0]]
        detected = [detected[-1], c.read_discrete_inputs(pin_detected)[0]]
        c_limit = [c_limit[-1], c.read_discrete_inputs(pin_c_limit)[0]]
        start = [start[-1], c.read_discrete_inputs(pin_start)[0]]
        reset = [reset[-1], c.read_discrete_inputs(pin_reset)[0]]
        stop = [stop[-1], c.read_discrete_inputs(pin_stop)[0]]
        emergency_stop = c.read_discrete_inputs(pin_emergency_stop)[0]
        x = [x[-1], c.read_input_registers(pin_read_x)[0]]
        y = [y[-1], c.read_input_registers(pin_read_y)[0]]
        z = [z[-1], c.read_input_registers(pin_read_z)[0]]
        
        print(f"[STATE] mode={mode} | tasks: T1={task1} T2={task2} T3={task3} T4={task4} T5={task5} T6={task6}")
        print(f"[POS]   X={x[-1]} Y={y[-1]} Z={z[-1]} | box@place={box_at_place[-1]} part@place={part_at_place[-1]}")
        
        if event(start)=="RE":    mode = "on"
        if event(stop)=="RE": mode = "off"
            
    
        if mode == "on" :
            
            """Executing the program"""
            #T1 : Bring the box with box_conveyor (5-1 --> T1 --> 1-4)
            if task5==True:
                if box_at_place[-1]==True :
                    box_conveyor = True
                elif event(box_at_place)=="FE": 
                    box_conveyor = False
                    exit_conveyor = False
                    task1 = True
                    task5 = False
                
            #T2 : Bring the part with part_conveyor (3-2 --> T2 --> 2-3)
            if task3==True or init==True:
                if part_at_place[-1] == True:
                    task2=10
                    part_conveyor = True
                elif event(part_at_place)=="FE":
                    part_conveyor = False                 
                    task2 = True
                    init = False
                    
                elif task2 == True and task4==True:
                    task3 = False
            
            #T3 : Take the box with the grap(6&2-3 --> T3 --> 3-4&2)
            if (task6==True and task2==True) :
                if z[-1] != 5 and grap == False:
                    spz = 6
                    task3 = 10
                elif z[-1] == 5 and grap==False and task3==10: 
                    grap=True
                    task3=20
                elif grap==True and z[-1] != 0 and task3==20:
                    spz = 0
                    task3 = 30
                elif z[-1]==0 and task3==30:
                    task3 = True
                    task6 = False
                    task2 = False
                
            #T4 : Put three part on the box (1&3-4 --> T4 --> 4-5&6)
            if task1==True and task3==True:
                if x[-1] != 3 and task4 == False:
                    spx = 3
                    task4=10
                elif x[-1] == 3 and z[-1] != 10 and task4==10:
                    spy = 6
                    spz = 10
                    task4=20
                elif y[-1] == spy-1 and z[-1] == 10 and grap==True and task4==20:
                    grap = False
                    task4 = 30
                elif grap == False and z[-1] != 0 and task4==30:
                    spz = 0                
                    task4 = 40
                elif task4 == 40 and z[-1]==0:
                    take_box = True
                    task3 = False
                    task4 = 50
                    
                #Start the second part 
                elif task4==50 and x[-1] != 3:
                    spx = 3
                    task4 = 60
                elif x[-1] == 3 and y[-1] != 3 and task4==60:
                    spy = 3
                    task4=70
                elif y[-1] == 3 and z[-1] != 10 and task4==70:
                    spz = 10
                    task4=75
                elif (y[-1] == spy) and z[-1] == 10 and grap==True and task4==75:
                    grap = False
                    task4 = 80
                elif grap == False and z[-1] != 0 and task4==80:
                    spz = 0                
                    task4 = 90
                elif task4 == 90 and z[-1]==0:
                    take_box = True
                    task3 = False
                    task4 = 100                    
        
                #Start the third part 
                elif task4==100 and x[-1] != 3:
                    c_plus = True
                    spx = 3
                    task4 = 110
                elif x[-1] == 3  and z[-1] != 5 and task4==110:
                    spz = 5
                    task4=120
                elif z[-1] == 5 and grap==True and task4==120:
                    grap = False
                    task4 = 130
                elif grap == False and z[-1] != 0 and task4==130:
                    c_plus = False
                    spz = 0                
                    task4 = 140
                elif task4 == 140 and z[-1]==0:
                    task3 = False
                    task4 = True     
                
            #T5 : Evacuate the box (4-5 --> T5 --> 5-1)
            if task4==True:
                if box_at_place[-1] == False:
                    exit_conveyor = True
                    box_conveyor = True
                elif event(box_at_place)=="RE": 
                    task5 = True
                    task4 = False
                    
            #T6 : Go to the box (4-6 --> T6 --> 6-3)
            if (task4==50 or task4==100 or task4==True or init) and task3==False:
                if x[-1]!=7 or (y[-1]!=4 and y[-1]!=5) or z[-1]!=0 :
                    spx = 8
                    spy = 5
                    spz = 0       
                    task6 = 10
                elif (x[-1]==7) and (y[-1]==4 or y[-1]==5) and (z[-1]==0):
                    task6 = True
                elif task5==True and task6==True:
                    task4 = False   
    
        elif mode=="off" :
            box_conveyor = False
            part_conveyor = False
            exit_conveyor = False
        
        """Writing the outputs"""
        c.write_single_coil(pin_box_conveyor, box_conveyor)
        c.write_single_coil(pin_part_conveyor, part_conveyor) 
        c.write_single_coil(pin_grap, grap) 
        c.write_single_coil(pin_c_plus, c_plus)
        
        c.write_single_register(pin_spx, spx)
        c.write_single_register(pin_spy, spy)
        c.write_single_register(pin_spz, spz)
        
        c.write_single_coil(pin_exit_conveyor, exit_conveyor)
        
        
        """Save the data"""
        observation = [box_at_place[-1], part_at_place[-1], detected[-1], c_limit[-1], start[-1], 
                       reset[-1], stop[-1], emergency_stop, box_conveyor, part_conveyor, 
                       exit_conveyor, c_plus, grap]
        observation = list(map(lambda x : 0 if x==False else 1 if x==True else x , observation))
        
        if observation != past_observation:
            past_observation = observation.copy()
            observation.append(time.time())
            with open(filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(observation)
                csvfile.close()