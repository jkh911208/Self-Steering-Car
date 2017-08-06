import os
import numpy as np
from random import shuffle
# os.path.exists("/etc/password.txt")
'''

# amount of lists you want to create
for i in range(0,16):
	plus = "" # this line is here to clear out the previous command
	plus = "plus" + str(i) + " = []"
	minus = "" # this line is here to clear out the previous command
	minus = "minus" + str(i) + " = []"
	exec(plus)
	exec(minus)

print(plus5)

for i in range(0,16):
	plus = "plus{}".format(i)
	print(list(plus))
'''
minus = []
plus =
for i in range(0,16):
	minus[i] = []
	plis[i] = []


print(minus, plus)

'''
plus_15 = []
plus_14 = []
plus_13 = []
plus_12 = []
plus_11 = []
plus_10 = []
plus_9 = []
plus_8 = []
plus_7 = []
plus_6 = []
plus_5 = []
plus_4 = []
plus_3 = []
plus_2 = []
plus_1 = []
plus_0 = []
minus_0 = []
minus_1 = []
minus_2 = []
minus_3 = []
minus_4 = []
minus_5 = []
minus_6 = []
minus_7 = []
minus_8 = []
minus_9 = []
minus_10 = []
minus_11 = []
minus_12 = []
minus_13 = []
minus_14 = []
minus_15 = []
'''

'''
print(data_set)

file_list = os.listdir('/raw_data')
#print(file_list)
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

for i in range(-20,21):
	if i == 0:
		print("data_set[-0] = ", data_set['-0'])
	string = str(i)
	print("data_set[{}] = ".format(string), data_set[string])



'''
