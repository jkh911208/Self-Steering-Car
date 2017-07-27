import can 
bus = can.interface.Bus(channel='can0', bustype='socketcan_native') 

while(True):
	notifier = str(bus.recv())
	if(notifier.find("ID: 0025") > 0):
		hex_data = notifier[-23:-21] + notifier[-20:-18]
		int_data = int(hex_data, 16)
		if(int_data == 0):
			pass
		elif(int_data > 500):
			int_data = int_data - 4096

		print(int_data)
