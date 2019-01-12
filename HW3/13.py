import numpy as np
from random import uniform,random
from utility import *

N = 1000
A = np.matrix([[1,uniform(-1, 1), uniform(-1, 1)] for _ in range(N)])
b = np.matrix([f(item[0,1], item[0,2]) * (1, -1)[random() < 0.1] for item in A]).T

w = (A.T * A).I * A.T * b
b_hat = A*w
print(err_num(b,b_hat,N)/N)