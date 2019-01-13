import sys
import random
import numpy as np


def load_data(file_path):
	X = []
	Y = []
	m = []
	with open(file_path) as f:
		for line in f:
			m.append([float(i) for i in line.split()])
	mm = np.array(m)
	row = len(m)
	X = np.c_[np.ones(row), mm[::, :-1]]
	Y = mm[:, -1]
	return X, Y



def test(X, Y, w) :
	n = len(Y)
	ne = sum([1 for i in range(n) if np.sign(np.dot(X[i], w)) != Y[i]])
	return ne# / float(n)

def train(X, Y, updates = 50, pocket = True) :
	col = len(X[0])
	n = len(X)
	w = np.zeros(col)
	wg = w
	error = test(X, Y, w);

	for k in range(updates) :
		idx = random.sample(range(n), n)
		for i in idx :
			if np.sign(np.dot(X[i], w)) != Y[i] :
				w = w + Y[i] * X[i]
				e = test(X, Y, w)
				#print(w,error)
				if e < error :
					error = e
					wg = w
				break
	if pocket :
		return wg
	return w

def main() :
	X, Y = load_data('hw1_18_train.dat')
	print(X[:10])
	TX, TY = load_data('hw1_18_test.dat')
	print(len(X),len(TX))
	error = 0
	n = 200
	for i in range(n) :
		w = train(X, Y, updates = 50)
		#w = train(X, Y, updates = 100)
		#w = train(X, Y, updates = 50, pocket = False)
		error += test(TX, TY, w)
	print(error / n/500)

if __name__ == '__main__' :
	main()
