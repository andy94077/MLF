'''read_data(filename), get_w(X, Y, lam), err_rate(X, Y, w)'''
import numpy as np
def read_data(filename):
	'''@return: matrix, matrix. X, Y'''
	data = np.loadtxt(filename)
	data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
	return np.matrix(data[:,:-1]), np.matrix(data[:, -1:])

def get_w(X, Y, lam):
	'''@return: array w'''
	return (X.T * X + lam * np.identity(X.shape[1])).I * X.T * Y

def err_rate(X, Y, w):
	'''@return: float'''
	return np.float((sum(np.sign(X * w) != Y) / X.shape[0]))
