from utility import *
from random import sample, seed, randint
import matplotlib.pyplot as plt

data = read_data('hw1_7_train.dat')

all_pass = False
test_time = 1126
update_distribution=[]
for t in range(test_time):
	w=np.zeros(5)
	update_n = 0
	seed(randint(-2147483648, 2147483647))
	all_pass = False
	while not all_pass:
		all_pass=True
		for i in sample(range(len(data)),len(data)):
			if np.sign(w.dot(data[i][0])) != data[i][1]:
				w += data[i][1] * data[i][0]
				all_pass = False
				update_n += 1
	update_distribution.append(update_n)
plt.hist(update_distribution)
plt.show()
		