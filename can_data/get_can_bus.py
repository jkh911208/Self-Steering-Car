import can
import numpy as np 
import signal
import sys

bus = can.interface.Bus(channel='can0', bustype='socketcan_native') 

data = []

def signal_handler(signal, frame):
	f.close()
	file_name = "slow.npy"
	np.save(file_name,data)
	sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

f = open("slow.txt", "w")

while(True):
	msg = str(bus.recv())
	if(msg.find("ID: 0025") > 0):
		f.write(msg)
		f.write('\n')
		data.append(msg)
		print(msg)
