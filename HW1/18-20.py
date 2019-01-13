from utility import *
from random import random, seed, choice, sample

def test_err(w, test_data):
	err = 0
	for v in test_data:
		if np.sign(w.dot(v[0])) != v[1]:
			err+=1
	return err


def PLA(train, test, update_time=50, pocket=True):
	err = 0
	test_time = 200
	for _ in range(test_time):
		w=np.zeros(5)
		new_w=np.zeros(5)
		seed(random())
		old_err = test_err(w,train)
		#training
		for t in range(update_time):
			for i in sample(range(len(train)),len(train)):
				v, y=train[i][0],train[i][1]
				if np.sign(new_w.dot(v)) != y:
					new_w += y * v
					if not pocket and t == update_time - 1:
						w = new_w.copy()
					new_err = test_err(new_w,train)
					if new_err < old_err:
						old_err = new_err
						w = new_w.copy()
					break
		#testing
		err += test_err(w, test)
	print('pocket: {0}, update time: {1}, error rate: {2}'.format(pocket, update_time, err/len(test)/test_time))
	
D, T = read_data('hw1_18_train.dat'), read_data('hw1_18_test.dat')
PLA(D, T)
PLA(D, T, pocket=False)
PLA(D, T, 100)
