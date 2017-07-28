import numpy as np 
import time

loaded_data = np.load("right_to_left.npy")

for i in loaded_data:
	msg = str(i)
	print(msg)
	hex_data = msg[-23:-21] + msg[-20:-18]
	int_data = int(hex_data, 16)

	if(int_data == 0):
		pass
	elif(int_data > 500):
		int_data = int_data - 4096

	print(int_data)
	
	time.sleep(0.5)

