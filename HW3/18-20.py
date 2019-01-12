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
	'''@return: normalized dE ndarray'''
	if sgd:
		dE=theta(-y[dE_norm.n] * w.dot(data[dE_norm.n])) * (-y[dE_norm.n] * data[dE_norm.n])
		dE_norm.n = (dE_norm.n + 1) % 50
	else:
		dE=sum([theta(-y[i] * w.dot(data[i])) * (-y[i] * data[i]) for i in sample(range(data.shape[0]),50)]) / 50
	return dE/np.linalg.norm(dE)
dE_norm.n=0

def PLA(data, y, test, eta, sgd=False, update_time=2000):
	'''@return: int. The number of errors after PLA'''
	w = np.zeros(data.shape[1])
	for _ in range(update_time):
		print('%d ' %_ if _%10==0 else '',end='')
		for i, x in enumerate(data):
			if sign(w.dot(x)) != y[i]:
				w -= eta * dE_norm(w, data, y,sgd)
		print('\n' if _%100==0 else '',end='')
	return sum([sign(w.dot(line[:-1]))!=line[-1] for line in test])


train, test = read_data('hw3_train.dat'), read_data('hw3_test.dat')
print('\n',PLA(train[:,:-1], train[:, -1], test, 0.001)/test.shape[0])
