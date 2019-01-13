'''np.sign(n), read_data(filename)'''
import numpy as np
def np.sign(n):
	return 1 if n > 0 else - 1

def read_data(filename):
	data = []
	with open(filename,'r') as f:
		for line in f:
			line = line.split()
			data.append((np.array([1]+list(map(float, line[:-1]))), int(line[-1])))
	return data
