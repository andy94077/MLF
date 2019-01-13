from random import random, randint, seed
from utility import *
train, test =read_data('hw1_18_train.dat'), read_data('hw1_18_test.dat')
def test_err(w, test_data):
	err = 0
	for v in test_data:
		if np.sign(w.dot(v[0])) != v[1]:
			err+=1
	return err

err = 0
test_time=2000
for _ in range(test_time):
	w=np.zeros(5)
	seed(random())
	w50 = None
	old_err=test_err(w,train)
	#training
	for t in range(50):
		i = randint(0, train.shape[0] - 1)
		if t == 49:
			w50 = w.copy()
		if np.sign(w.dot(train[i][0])) != train[i][1]:
			new_w = w + train[i][1] * train[i][0]
			if t == 49:
				w50 = new_w.copy()
				break
			new_err=test_err(new_w,train)
			if new_err < old_err:
				w += train[i][1] * train[i][0]
	#testing
	err+=test_err(w50,test)

print(err/test.shape[0]/test_time,)
		