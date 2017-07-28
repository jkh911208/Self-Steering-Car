import numpy as np 
import time

f = open("right_to_left.txt","w")
loaded_data = np.load("right_to_left.npy")

for i in loaded_data:
	msg = i
	f.write(msg)
	f.write('\n')
	print(msg)
	#time.sleep(0.5)

f.close()
