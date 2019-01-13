import numpy as np
import matplotlib.pyplot as plt
def read_data(filename):
	'''@return: data array. Therefore data[i,:-1] is the line data, and data[i,-1] is the result y'''
	return np.loadtxt(filename)

def theta(s):
	'''@retrun: double'''
	return 1 / (1 + np.exp(-s))

def dE_norm(w, data, y,n):
	'''@return: dE ndarray'''
	if n >= 0:
		dE = theta(-y[n,0] * data[n].dot(w)) * (-y[n,0] * data[n])
	else:
		dE = np.sum(theta(-y * (data.dot(w))) * (-y * data) / data.shape[0], axis=0)
	return dE.reshape(-1,1)

def err_rate(w, data, y):
	return np.sum(np.sign(data.dot(w))!=y)/data.shape[0] 

def PLA(data, y, test, eta, sgd=False, update_time=2000):
	'''@return: array, array. Ein, E_out'''
	w = np.zeros((data.shape[1],1))
	ein_list = np.empty(update_time)
	eout_list = np.empty(update_time)

	n = 0 if sgd else - 1
	for t in range(update_time):
		w -= eta * dE_norm(w, data, y, n)
		n = (n + 1) % data.shape[0] if sgd else -1
		
		ein_list[t]=err_rate(w, data, y)
		eout_list[t]=err_rate(w,test[:,:-1], test[:, -1:])
	return ein_list, eout_list


update_time=2000
train, test = read_data('hw3_train.dat'), read_data('hw3_test.dat')

GD001 = PLA(train[:,:-1], train[:, -1:], test, 0.01)
SGD0001 = PLA(train[:,:-1], train[:, -1:], test, 0.001, True)
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

