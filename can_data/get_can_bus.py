import can
import numpy as np 
import signal
import sys

bus = can.interface.Bus(channel='can0', bustype='socketcan_native') 

data = []

def signal_handler(signal, frame):
        file_name = "right_to_left.npy"
        np.save(file_name,data)
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while(True):
	msg = str(bus.recv())
	if(msg.find("ID: 0025") > 0):
		data.append(msg)
		print(msg)
