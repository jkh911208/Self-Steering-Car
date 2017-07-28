import numpy as np
import os
import cv2

from networks import googLeNet

WIDTH = 256
HEIGHT = 66
LR = 1e-2
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('googLeNet', LR , EPOCHS)

model = googLeNet(HEIGHT,WIDTH, LR)

# start training
file_list = os.listdir('/raw_data')
for file in file_list:
	if file.endswith('.npy'):
		print("Start training on file : " + file)
		file_location = "/raw_data/" + file
		loaded_data = np.load(file_location)

		# change the saved data into the form that both human and NN can understand
		for data in loaded_data:
			# flip the frame because the webcam is mounted upside down on the front windsheild, and cut out the sky portion of the image the size became 256*66
			tmp = cv2.flip(data[0],0)
			tmp = cv2.flip(tmp,1)
			data[0] = (tmp[70:-5,::]).reshape(66,256,1)


			'''
			Example of interpreting the can steering data
			1) left:
				Timestamp: 1501251285.664896        ID: 0025    000    DLC: 8    
				00 03 30 01 bf ce 00 ee
				[-23:-21][-20:-18][-17:-15][-14:-12][-11:-9][-8:-7][-6:-4][-3:-1]
				first two bytes are the int before decimal and last one byte is decimal
			'''

			# change the can data (HEX) to numerical data
			tmp = data[1]
			hex_data = tmp[-23:-21] + tmp[-20:-18]
			hex_decimal = tmp[-3:-1]
			int_data = int(hex_data, 16)
			int_decimal = int(hex_decimal, 16)
			
			# if the steering wheel angle in in right to the center
			if(int_data > 550):
				int_data = int_data - 4096
				int_decimal = 256 - int_decimal 
			
			# put the int and the decimal together
			num_in_string = str(int_data) + "." + str(int_decimal)
			final_data = float(num_in_string)
			data[1] = final_data
		
		train = loaded_data[:-100]
		test = loaded_data[-100:]

		train_X = np.array([i[0] for i in train]).reshape(-1, HEIGHT,WIDTH,1)
		train_Y = np.array([i[1] for i in train]).reshape(-1,1)

		val_X = np.array([i[0] for i in test]).reshape(-1, HEIGHT,WIDTH,1)
		val_Y = np.array([i[1] for i in test]).reshape(-1,1)


		
		model.fit(train_X, train_Y, n_epoch=EPOCHS, run_id=MODEL_NAME, show_metric=True, validation_set=(val_X,val_Y))

		model.save(MODEL_NAME)
