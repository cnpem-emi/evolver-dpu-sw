import json
import socket
import random
import time
from template.consts import functions

STIR_VALUES = [[0] * 16, [4000] * 16] # flip between on and off
TEMP_VALUES = [4095] * 16 # always off

EVOLVER_IP = '10.0.6.69'
EVOLVER_PORT = 6001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((EVOLVER_IP, EVOLVER_PORT))   

def run_test(time_to_wait, selection):
	time.sleep(time_to_wait)
	print('Sending data...')

	# Send temp	
	data = {'param': 'temp', 'value': [4095] * 16, 'immediate': True}
	print(data)
	s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')

	# Send stir
	data = {'param': 'stir', 'value': STIR_VALUES[selection], 'immediate': True}
	print(data)
	s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')

	# Run commands
	s.send(functions['run_commands']['id'].to_bytes(1,'big') + b'\r\n')

	# Set things for the next one
	selection = 1 - selection
	time_to_wait = random.randint(1, 31)
	print('Seconds to wait: ' + str(time_to_wait))
	run_test(time_to_wait, selection)


if __name__ == '__main__':

	time.sleep(5)
	run_test(0, 0)

