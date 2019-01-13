#!/usr/bin/env python3
from random import uniform, random
import matplotlib.pyplot as plt
import numpy as np
def h(s, x, theta):
	return s * np.sign(x - theta)
def test(data, y, s, theta):
	'''return: the number of errors that h(s,data[i],theta) != y[i]'''
	return sum(1 if h(s,x,theta)!=y[i] else 0 for i,x in enumerate(data))

def error(N):
	D = [uniform(-1, 1) for _ in range(N)]
	y = [np.sign(i) if random()>0.8 else -np.sign(i) for i in D]

	#add a virtual head data and virtual tail data to D for the purpose of choosing mean easily
	D.insert(0, D[0] - 1)
	D.append(D[-1] + 1)

	ein = N
	min_s ,min_theta = 0,0.0
	for s in [-1, 1]:
		for i in range(N+1):
			theta = (D[i] + D[i + 1]) / 2 # choose mean as theta
			new_ein = test(D[1:-1], y, s, theta) # D[1:-1] ignores the virtual head and the virtual tail
			if new_ein < ein:
				ein = new_ein
				min_s = s
				min_theta = theta
	return ein/N - 0.5+0.3 * min_s * (abs(min_theta) - 1) #ein-eout

N=20
loop_times=1000
ein_minus_eout = [error(N) for _ in range(loop_times)]

plt.hist(ein_minus_eout,bins=15)
plt.title('Ein-Eout')
plt.show()
