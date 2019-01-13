from utility import *
from random import sample
data=read_data('hw1_15_train.dat')

w = np.zeros(5)
all_pass = False
update_n = 0
alpha=0.5
israndom = True
if israndom:
	test_time=2000
	for t in range(test_time):
		w=np.zeros(5)
		all_pass = False
		while not all_pass:
			all_pass=True
			for i in sample(range(len(data)),len(data)):
				if np.sign(w.dot(data[i][0])) != data[i][1]:
					w += alpha*data[i][1] * data[i][0]
					all_pass = False
					update_n += 1
	print(update_n/test_time, w)
else:
	while not all_pass:
		all_pass=True
		for v in data:
			if np.sign(w.dot(v[0])) != v[1]:
				w += v[1] * v[0]
				all_pass = False
				update_n+=1
	print(update_n, w)
		