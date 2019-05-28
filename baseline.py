from sklearn.neural_network import MLPRegressor

def nn_model(X_train, X_test, y_train, lr=0.001, max_iter=1600, layers=(64, 128, 100)):
    model = MLPRegressor(learning_rate_init=lr, max_iter=max_iter, hidden_layer_sizes = layers)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred