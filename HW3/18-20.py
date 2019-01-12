import numpy as np
from math import exp
from random import sample
from utility import *

def theta(s):
	'''@retrun: double'''
	return 1 / (1 + exp(-s))

def dE_norm(w, data, y):
	'''@return: normalized dE ndarray'''
	dE=sum([theta(-y[i] * w.dot(data[i])) * (-y[i] * data[i]) for i in sample(range(data.shape[0]),50)]) / data.shape[0]
	return np.linalg.norm(dE)
	
def PLA(data, y, test, eta, update_time):
	'''@return: int. the number of errors after PLA'''
	w = np.zeros(data.shape[1])
	for _ in range(update_time):
		print('%d ' %_ if _%10==0 else '',end='')
		for i, x in enumerate(data):
			if sign(w.dot(x)) != y[i]:
				w -= eta * dE_norm(w, data, y)
		print('\n' if _%100==0 else '',end='')
	return sum([sign(w.dot(line[:-1]))!=line[-1] for line in test])
	
train, test = read_data('hw3_train.dat'), read_data('hw3_test.dat')
print(PLA(train[:,:-1], train[:, -1], test, 0.001, 2000))
