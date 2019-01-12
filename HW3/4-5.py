import numpy as np
from math import exp
from random import sample
import matplotlib.pyplot as plt
from utility import *

def theta(s):
	'''@retrun: double'''
	if s < -20:
		return 0
	elif s > 20:
		return 1
	else:
		return 1 / (1 + exp(-s))

def dE_norm(w, data, y,n):
	'''@return: dE ndarray'''
	if n >= 0:
		dE=theta(-y[n] * w.dot(data[n])) * (-y[n] * data[n])
	else:
		dE=sum([theta(-y[i] * w.dot(data[i])) * (-y[i] * data[i]) for i in range(data.shape[0])]) / data.shape[0]
	return dE

def err_rate(w, data, y):
	return sum([sign(w.dot(line[0]))!=line[1] for line in zip(data,y)])/data.shape[0]

def PLA(data, y, test, eta, sgd=False, update_time=2000):
	'''@return: int, int. Ein, E_out'''
	w = np.zeros(data.shape[1])
	ein_list = []
	eout_list = []

	n = 0 if sgd else - 1
	for _ in range(update_time):
		w -= eta * dE_norm(w, data, y, n)
		n = (n + 1) % data.shape[0] if sgd else -1
		
		ein_list.append(err_rate(w, data, y))
		eout_list.append(err_rate(w,test[:,:-1], test[:, -1]))
	return ein_list, eout_list


update_time=2000
train, test = read_data('hw3_train.dat'), read_data('hw3_test.dat')

GD001 = PLA(train[:,:-1], train[:, -1], test, 0.01)
SGD0001 = PLA(train[:,:-1], train[:, -1], test, 0.001, True)
print('GD, eta:0.01, Ein: {}, Eout: {}'.format(GD001[0][-1], GD001[1][-1]))
print('SGD, eta:0.001, Ein: {}, Eout: {}'.format(SGD0001[0][-1], SGD0001[1][-1]))

plt.title('Ein')
plt.ylabel('Ein(w_t)')
plt.xlabel('t')
plt.plot(range(update_time), GD001[0],label='GD')
plt.plot(range(update_time), SGD0001[1], label='SGD')
plt.legend()
plt.show()

plt.title('Eout')
plt.ylabel('Eout(w_t)')
plt.xlabel('t')
plt.plot(range(update_time), GD001[0],label='GD')
plt.plot(range(update_time), SGD0001[1], label='SGD')
plt.legend()
plt.show()

