'''np.sign(x), f(x1,x2), read_data(filename), err_rate(b,b_hat,N)'''
import numpy as np
def f(x1,x2):
	return np.sign(x1 ** 2 + x2 ** 2 - 0.6)

def read_data(filename):
	'''@return: data array. Therefore data[i,:-1] is the line data, and data[i,-1] is the result y'''
	data = []
	with open(filename,'r') as f:
		data = [list(map(float, line.split())) for line in f]
	return np.array(data)


def err_num(b,b_hat,N):
	'''@return: the num of error that b[i] != b_hat[i]'''
	b = np.array(b).flat
	b_hat=np.array(b_hat).flat
	return sum([ b[i]!=np.sign(b_hat[i]) for i in range(N)])
