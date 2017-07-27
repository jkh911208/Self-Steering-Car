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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # webcam is install in car upside down, so flip the image
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,1)

    # cut the top part of image because it is sky and bottom of image
    # frame = frame[150:-15,::]

    # resize the image by 80% to make it right size to feed into CNN
    frame = cv2.resize(frame,(0,0),fx=0.8, fy=0.8)

    # print(frame.shape)
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    return frame

def get_angle():
    while(True):
        notifier = str(bus.recv())

        if(notifier.find("ID: 0025") > 0):
            hex_data = notifier[-23:-21] + notifier[-20:-18]
            int_data = int(hex_data, 16)

            if(int_data > 550):
                int_data = int_data - 4096

            break

    return int_data

def main():
	while True:
		# discard first 40 frames to give time for webcam to get the proper exposure
		for _ in range(40):
			get_angle()
			get_frame()

		angle = get_angle()
		frame = get_frame()
		raw_data.append([frame, angle])
		print(frame.shape[0], frame.shape[1], angle)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break   

main()

cap.release()
# cv2.destroyAllWindows()
