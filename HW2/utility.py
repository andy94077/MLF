'''sign(x), h(s, x, theta), read_data(filename), test(data, y, s, theta)'''
import numpy as np
def sign(x):
	return (-1,1)[x>0]
def h(s, x, theta):
	return s * sign(x - theta)

def read_data(filename):
	'''@return: transposed data array. Therefore data[:0] is the original line data, and data[-1] are the result y'''
	data = []
	with open(filename,'r') as f:
		data = [list(map(float, line.split())) for line in f]
	return np.array(data).T


def test(data, y, s, theta):
	'''@return: the number of errors that h(s,data[i],theta) != y[i]'''
	return sum((0,1)[h(s,x,theta)!=y[i]] for i,x in enumerate(data))
