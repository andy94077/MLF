from utility import *

trainX, trainY = read_data('hw4_train.dat')
testX, testY = read_data('hw4_test.dat')

w = get_w(trainX, trainY, 10)
print('lamda: 10, Ein:', err_rate(trainX, trainY, w))
print('lamda: 10, Eout:', err_rate(testX, testY, w))

