import os
import numpy as np
import time
from random import shuffle

data_size = 512
minus = []
plus = []
for i in range(0,16):
	minus.append([])
	plus.append([])

file_list = os.listdir('/raw_data')
#print(file_list)
for file_name in file_list:
	if file_name.endswith('.npy'):
		print(file_name)
		file_location = "/raw_data/" + file_name
		loaded_data = np.load(file_location)
		loaded_data = loaded_data[30:]

		for data in loaded_data:
			# change the can data (HEX) to numerical data
			tmp = data[1]
			hex_data = tmp[-23:-21] + tmp[-20:-18]
			hex_decimal = tmp[-3:-1]
			int_data = int(hex_data, 16)
			int_decimal = int(hex_decimal, 16) / 256

			# if the steering wheel angle in in right to the center
			if(int_data > 550):
				int_data = int_data - 4095
				int_decimal = 1 - int_decimal 
				steering_angle = int_data - int_decimal
			else:
				# put the int and the decimal together
				steering_angle = int_data + int_decimal
			
			int_data = int(steering_angle)

			if int_data >= 16 or int_data <= -16:
				continue

			if int_data == 0:
				if int_data > steering_angle:
					minus[int_data].append([data[0], data[1]])
				else:
					plus[int_data].append([data[0], data[1]])
			elif int_data > 0:
				plus[int_data].append([data[0], data[1]])
			else:
				int_data = int_data * -1
				minus[int_data].append([data[0], data[1]])

for i in range(0,16):
	print("minus[{}] :".format(i),len(minus[i]),"plus[{}] :".format(i), len(plus[i]))


for i in range(0,16):
	shuffle(minus[i])
	shuffle(plus[i]) 

	minus[i] = minus[i][:data_size]
	plus[i] = plus[i][:data_size]


for i in range(0,16):
	print("minus[{}] :".format(i),len(minus[i]),"plus[{}] :".format(i), len(plus[i]))


for i in range(0,16):
	if len(minus[i]) != data_size:
		while True:
			minus[i] = minus[i] + minus[i] 
			if len(minus[i]) >= data_size:
				break

	if len(plus[i]) != data_size:
		while True:
			plus[i] = plus[i] + plus[i] 
			if len(plus[i]) >= data_size:
				break
for i in range(0,16):
	print("minus[{}] :".format(i),len(minus[i]),"plus[{}] :".format(i), len(plus[i]))

for i in range(0,16):
	shuffle(minus[i])
	shuffle(plus[i]) 

	minus[i] = minus[i][:data_size]
	plus[i] = plus[i][:data_size]

for i in range(0,16):
	print("minus[{}] :".format(i),len(minus[i]),"plus[{}] :".format(i), len(plus[i]))


batch_size = 64
batch = int(data_size / batch_size)
#print(data_size/batch_size , batch)

for j in range(batch):
	temp_list = []
	for i in range(0,16):
		temp_list = temp_list + minus[i][j*batch_size:(j+1)*batch_size]
		#print("temp_minus : " , len(temp_minus))
		temp_list = temp_list + plus[i][j*batch_size:(j+1)*batch_size]
		#print("temp_plus : ", len(temp_plus))

	name = str(time.time())
	file_name = "/raw_data/balanced_data/" + name + ".npy"
	print(len(temp_list))
	shuffle(temp_list)
	np.save(file_name,temp_list)
	print("Saved file name : ",name + ".npy")

