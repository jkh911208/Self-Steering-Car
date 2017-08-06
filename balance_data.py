import os
import numpy as np
from random import shuffle

data_set = {'-0': 0}
for i in range(-20,21):
	string = str(i)
	data_set[string] = 0

print(data_set)

file_list = os.listdir('/raw_data')
print(file_list)
file_list = list(file_list)
print(file_list)
file_list = shuffle(list(file_list))
print(file_list)
for file_name in file_list:
	if file_name.endswith('.npy'):
		# print(file_name)
		file_location = "/raw_data/" + file_name
		loaded_data = np.load(file_location)


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
				final_data = int_data - int_decimal
			else:
				# put the int and the decimal together
				final_data = int_data + int_decimal
			
			int_data = int(final_data)
			if int_data == 0:
				if int_data > final_data:
					data_set['-0'] = data_set['-0'] + 1
				else:
					data_set['0'] = data_set['0'] + 1
			else:
				string = str(int_data)
				# print(string)
				data_set[string] = data_set[string] + 1

print(data_set)




