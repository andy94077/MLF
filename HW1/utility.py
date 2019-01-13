'''np.sign(n), read_data(filename)'''
import numpy as np
def read_data(filename):
	data = []
	with open(filename,'r') as f:
		for line in f:
			line = line.split()
			data.append((np.array([1]+list(map(float, line[:-1]))), int(line[-1])))
	return np.array(data)
