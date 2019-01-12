import numpy as np
from math import exp
from random import sample
from utility import *

def theta(s):
	'''@retrun: double'''
	if s < -20:
		return 0
	elif s > 20:
		return 1
	else:
		return 1 / (1 + exp(-s))

def dE_norm(w, data, y,sgd):
	'''@return: dE ndarray'''
	if sgd:
		dE=theta(-y[dE_norm.n] * w.dot(data[dE_norm.n])) * (-y[dE_norm.n] * data[dE_norm.n])
		dE_norm.n = (dE_norm.n + 1) % data.shape[0]
	else:
		dE=sum([theta(-y[i] * w.dot(data[i])) * (-y[i] * data[i]) for i in range(data.shape[0])]) / data.shape[0]
	return dE
dE_norm.n=0

def PLA(data, y, test, eta, sgd=False, update_time=2000):
	'''@return: int. The number of errors after PLA'''
	w = np.zeros(data.shape[1])
	for _ in range(update_time):
		w -= eta * dE_norm(w, data, y,sgd)
	return sum([sign(w.dot(line[:-1]))!=line[-1] for line in test])


train, test = read_data('hw3_train.dat'), read_data('hw3_test.dat')
print('\nEout: ',PLA(train[:,:-1], train[:, -1], test, 0.001)/test.shape[0])
