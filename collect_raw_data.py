import numpy as np
import cv2
import time
import os
import can
import signal
import sys

bus = can.interface.Bus(channel='can0', bustype='socketcan_native')

cap = cv2.VideoCapture(0)
cap.set(3,300) # set height
cap.set(4,200) # set width

raw_data = []

def signal_handler(signal, frame):
	name = str(time.time())
	file_name = "/raw_data/" + name + ".npy"
	np.save(file_name,raw_data)
	print("saved {} frame, name : {}".format(len(raw_data),name))
	sys.exit(0)
 
signal.signal(signal.SIGINT, signal_handler)

def get_frame():
	# read the frame from webcam
	_, frame = cap.read()

	# change the color to gray
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# change the resolution (256*141), later cut the sky
	frame = cv2.resize(frame,(0,0),fx=0.8, fy=0.8)
	
	return frame

def get_angle():
    while(True):
	# get one can data
        can_data = str(bus.recv())
	# check it is steering angle, if yes then stop the loop
        if(can_data.find("ID: 0025") > 0):
            break

    return can_data

def main():
	global raw_data

	# discard first 40 frames to give time for webcam to get the proper exposure
	for i in range(40):
			get_angle()
			get_frame()
			print(i, "discarded")

	while True:
		angle = get_angle()
		frame = get_frame()
		raw_data.append([frame, angle])
		
		if len(raw_data) % 1000 == 0:
			print(len(raw_data))
		'''		
		if len(raw_data) == 5000:
			pid = os.fork()
			if pid == 0:
				name = str(time.time())
				file_name = "/raw_data/" + name + ".npy"
				np.save(file_name,raw_data)
				print("saved {} frame, name : {}".format(len(raw_data),name))
				exit(0)
			else:
				raw_data = []
		'''			

main()

cap.release()
# cv2.destroyAllWindows()
