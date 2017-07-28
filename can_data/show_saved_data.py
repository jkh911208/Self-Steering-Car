import numpy as np 
import time

f = open("left_to_right.txt","w")
loaded_data = np.load("left_to_right.npy")

for i in loaded_data:
	msg = i
	f.write(msg)
	f.write('\n')
	print(msg)
	#time.sleep(0.5)

f.close()
