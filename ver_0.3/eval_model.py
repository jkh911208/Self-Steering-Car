import numpy as np
import os

from networks import googLeNet

WIDTH = 66
HEIGHT = 256
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('googLeNet', LR , EPOCHS)

model = googLeNet(WIDTH, HEIGHT, LR)


# make evaluation data
eval_x = []
eval_y = []
file_list = os.listdir('/raw_data/eval_data')
for file in file_list:
	if file.endswith('.npy'):
		file_location = "/raw_data/eval_data/" + file
		file = np.load(file_location)

		temp_x = []
	
		for data in file:
			frame = (data[0][70:-5,::]).reshape(66,256,1)
			#cv2.imshow('frame', frame)
			

			if len(temp_x) == 0:
				temp_x = frame
				#print(temp_x.shape)
			elif temp_x.shape[2] < 4:
				temp_x = np.concatenate((temp_x, frame), axis=2)
				#print(temp_x.shape)
			else:
				if temp_x.shape[2] == 4:
					temp_x = np.concatenate((temp_x, frame), axis=2)
					#print(temp_x.shape)
					eval_x.append([temp_x])
					eval_y.append([data[1]])
				else:
					temp_x = temp_x[::,::,-4:]
					#print(temp_x.shape)
					temp_x = np.concatenate((temp_x, frame), axis=2)
					#print(temp_x.shape)
					eval_x.append([temp_x])
					eval_y.append([data[1]])

eval_x = np.array(eval_x).reshape(-1,66,256,5)
#print(eval_x.shape)
eval_y = np.array(eval_y)
#print(eval_y.shape)
# finish makeing evaluation data

model.load(MODEL_NAME)

eval = model.evaluate(eval_x, eval_y)

print(eval)




