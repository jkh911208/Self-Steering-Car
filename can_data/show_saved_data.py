import numpy as np 
import time

f = open("test.txt","w")
loaded_data = np.load("left_to_right.npy")

for i in loaded_data:
	msg = i
	f.write(msg)
	print(msg)
	#time.sleep(0.5)

f.close()
