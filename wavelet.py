import pywt
import numpy as np
from sklearn.neural_network import MLPRegressor

def get_components(X):
	coeffs = pywt.wavedec(X, 'db1', level = 4)
	return coeffs

def fit_model(c1, c2, c3, c4, c5, c6, c7, y, lr=0.001, max_iter=800, layers=(64, 128, 100)):
	train =  np.array([c1, c2, c3, c4, c5, c6, c7])
	train = np.transpose(train)
	model = MLPRegressor(learning_rate_init=lr, max_iter=max_iter, hidden_layer_sizes = layers)
	model.fit(train, y)
	return model

def predict_model(c1, c2, c3, c4, c5, c6, c7, model):
	test = np.array([c1, c2, c3, c4, c5, c6, c7])
	test = np.transpose(test)
	y = model.predict(test)
	return y

def wavelet_model(X_train, X_test, y_train, lr=0.001, max_iter=800, layers=(64, 128, 100)):
	cA4_1, cD4_1, cD3_1, cD2_1, cD1_1 = get_components(X_train[:,0])
	cA4_2, cD4_2, cD3_2, cD2_2, cD1_2 = get_components(X_train[:,1])
	cA4_3, cD4_3, cD3_3, cD2_3, cD1_3 = get_components(X_train[:,2])
	cA4_4, cD4_4, cD3_4, cD2_4, cD1_4 = get_components(X_train[:,3])
	cA4_5, cD4_5, cD3_5, cD2_5, cD1_5 = get_components(X_train[:,4])
	cA4_6, cD4_6, cD3_6, cD2_6, cD1_6 = get_components(X_train[:,5])
	cA4_7, cD4_7, cD3_7, cD2_7, cD1_7 = get_components(X_train[:,6])

	cA4_y, cD4_y, cD3_y, cD2_y, cD1_y = get_components(y_train)

	model1 = fit_model(cA4_1, cA4_2, cA4_3, cA4_4, cA4_5, cA4_6, cA4_7, cA4_y, lr, max_iter, layers)
	model2 = fit_model(cD4_1, cD4_2, cD4_3, cD4_4, cD4_5, cD4_6, cD4_7, cD4_y, lr, max_iter, layers)
	model3 = fit_model(cD3_1, cD3_2, cD3_3, cD3_4, cD3_5, cD3_6, cD3_7, cD3_y, lr, max_iter, layers)
	model4 = fit_model(cD2_1, cD2_2, cD2_3, cD2_4, cD2_5, cD2_6, cD2_7, cD2_y, lr, max_iter, layers)
	model5 = fit_model(cD1_1, cD1_2, cD1_3, cD1_4, cD1_5, cD1_6, cD1_7, cD1_y, lr, max_iter, layers)

	cA4_1, cD4_1, cD3_1, cD2_1, cD1_1 = get_components(X_test[:,0])
	cA4_2, cD4_2, cD3_2, cD2_2, cD1_2 = get_components(X_test[:,1])
	cA4_3, cD4_3, cD3_3, cD2_3, cD1_3 = get_components(X_test[:,2])
	cA4_4, cD4_4, cD3_4, cD2_4, cD1_4 = get_components(X_test[:,3])
	cA4_5, cD4_5, cD3_5, cD2_5, cD1_5 = get_components(X_test[:,4])
	cA4_6, cD4_6, cD3_6, cD2_6, cD1_6 = get_components(X_test[:,5])
	cA4_7, cD4_7, cD3_7, cD2_7, cD1_7 = get_components(X_test[:,6])

	y1 = predict_model(cA4_1, cA4_2, cA4_3, cA4_4, cA4_5, cA4_6, cA4_7, model1)
	y2 = predict_model(cD4_1, cD4_2, cD4_3, cD4_4, cD4_5, cD4_6, cD4_7, model2)
	y3 = predict_model(cD3_1, cD3_2, cD3_3, cD3_4, cD3_5, cD3_6, cD3_7, model3)
	y4 = predict_model(cD2_1, cD2_2, cD2_3, cD2_4, cD2_5, cD2_6, cD2_7, model4)
	y5 = predict_model(cD1_1, cD1_2, cD1_3, cD1_4, cD1_5, cD1_6, cD1_7, model5)

	y_pred = [y1, y2, y3, y4, y5]
	y_pred = pywt.waverec(y_pred, 'db1')

	return y_pred