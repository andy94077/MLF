from utility import *

def get_w_by_lamdas(trainX, trainY, testX=None, testY=None, lams=range(-10, 3)):
	'''@return: float, float, array. min_err_lam, err , min_err_w'''
	min_err_lam = None
	min_err_w = None
	err = 1.0
	for lam in lams:
		w = get_w(trainX, trainY, 10.0 ** lam)
		new_err = err_rate(trainX, trainY, w) if testX is None or testY is None else err_rate(testX, testY, w)
		if new_err <= err:
			min_err_lam = lam
			min_err_w = w.copy()
			err = new_err
	return min_err_lam, err , min_err_w

trainX, trainY = read_data('hw4_train.dat')
testX, testY = read_data('hw4_test.dat')


#14
min_ein_lam, ein,min_ein_w = get_w_by_lamdas(trainX, trainY)
print('min ein, log10(lamda): {}, Ein: {}, Eout: {}'.format(min_ein_lam, ein, err_rate(testX, testY, min_ein_w)))

#15
min_eout_lam, eout, min_eout_w = get_w_by_lamdas(trainX, trainY, testX, testY)
print('min eout, log10(lamda): {}, Ein: {}, Eout: {}'.format(min_eout_lam, err_rate(trainX, trainY, min_eout_w), eout))

#16
min_etrain_lam, etrain, min_etrain_w = get_w_by_lamdas(trainX[:120], trainY[:120])
print('min etrain, log10(lamda): {}, Etrain: {}, Eval: {}, Eout: {}'.format(min_etrain_lam, etrain, err_rate(trainX[120:], trainY[120:], min_etrain_w), err_rate(testX, testY, min_etrain_w)))

#17
min_eval_lam, Eval, min_eval_w = get_w_by_lamdas(trainX[:120], trainY[:120], trainX[120:], trainY[120:])
print('min eval, log10(lamda): {}, Etrain: {}, Eval: {}, Eout: {}'.format(min_eval_lam, err_rate(trainX[:120], trainY[:120], min_eval_w), Eval, err_rate(testX, testY, min_eval_w)))

#18
min_eval_lam, Eval, min_eval_w = get_w_by_lamdas(trainX[:120], trainY[:120], trainX[120:], trainY[120:])
w = get_w(trainX, trainY, min_eval_lam)
print('min ein, Ein: {}, Eout: {}'.format(err_rate(trainX, trainY, w), err_rate(testX, testY, w)))

