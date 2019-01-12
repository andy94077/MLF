import numpy as np
from random import uniform,random
from utility import *

N = 1000
A = []
for _ in range(N):
	x1 = uniform(-1, 1)
	x2 = uniform(-1, 1)
	A.append([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
A = np.matrix(A)
b = np.matrix([f(item[0,1], item[0,2]) * (1, -1)[random() < 0.1] for item in A]).T

w = (A.T * A).I * A.T * b
b_hat = A * w
#print(w)
print(err_num(b,b_hat,N)/N)