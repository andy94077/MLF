from random import uniform, shuffle, random
from utility import *
def error(N):
	D = [uniform(-1, 1) for _ in range(N)]
	y = [np.sign(x) if random()>0.8 else -np.sign(x) for i, x in enumerate(D)]

	D.insert(0, D[0] - 1)
	D.append(D[-1] + 1)

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
	return ein, 0.5+0.3 * min_s * (abs(min_theta) - 1)

N=20
total_ein = ave_eout = 0.0
loop_times=1000
for _ in range(loop_times):
	ret = error(N)
	#print(ret[0],ret[1])
	total_ein += ret[0]
	ave_eout += ret[1]

print(total_ein/N/loop_times, ave_eout/loop_times)
