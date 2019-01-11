from utility import *
def error(data,y):
	'''@return: (ein,min_s,min_theta)'''
	N=data.shape[0]
	D = [data[0] - 1] + list(data) + [data[-1] + 1]
	
	ein = N
	min_s ,min_theta = 0,0.0
	for s in [-1, 1]:
		for i in range(N+1):
			theta = (D[i] + D[i + 1]) / 2
			new_ein = test(D[1:-1], y, s, theta)
			if new_ein < ein:
				ein = new_ein
				min_s = s
				min_theta = theta
	return ein,min_s,min_theta

def min_ein(dim_min_ein):
	min_i,min_ein=0,dim_min_ein[0]
	for i,item in enumerate(dim_min_ein):
		if item[0] < min_ein[0]:
			min_i = i
			min_ein = item
	return min_i,min_ein

D, test_data = read_data('hw2_train.dat'), read_data('hw2_test.dat')

dim_min_ein=[error(dim_data, D[-1]) for dim_data in D[:-1]]
all_min_i, all_min_ein = min_ein(dim_min_ein)
total_eout = test(test_data[all_min_i], test_data[-1], all_min_ein[1], all_min_ein[2])

print(all_min_ein[0] / D.shape[1], total_eout / test_data.shape[1])
