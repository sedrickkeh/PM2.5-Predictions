from sklearn.cluster import KMeans
from baseline import nn_model

def cluster_split(X, y, labels, clusters, istrain):
	X_clusters = []
	y_clusters = []
	for i in range(clusters):
		X_clusters.append([])
		y_clusters.append([])

	for i in range(len(labels)):
		X_clusters[labels[i]].append(X[i])
		y_clusters[labels[i]].append(y[i])

	if istrain:
		return X_clusters, y_clusters
	else:
		return X_clusters


def get_y_test_clusters(X_train_clusters, X_test_clusters, y_train_clusters, lr, max_iter, layers):
	y_test_clusters = []
	for i in range(len(X_train_clusters)):
		y_test = nn_model(X_train_clusters[i], X_test_clusters[i], y_train_clusters[i], lr, max_iter, layers)
		y_test_clusters.append(y_test)
	return y_test_clusters


def combine_preds(y_test_clusters, labels):
	preds = []
	cluster_counters = []
	for i in range(len(y_test_clusters)):
		cluster_counters.append(0)
	for i in labels:
		preds.append(y_test_clusters[i][cluster_counters[i]])
		cluster_counters[i] += 1
	return preds


def cluster_model(X_train, X_test, y_train, lr=0.001, max_iter=1600, layers=(64, 128, 100), clusters = 3):
	kmeans = KMeans(n_clusters=clusters).fit(X_train)
	X_train_clusters, y_train_clusters = cluster_split(X_train, y_train, kmeans.labels_, clusters, True)
	new_labels = kmeans.predict(X_test)
	X_test_clusters = cluster_split(X_test, y_train, new_labels, clusters, False)

	y_test_clusters = get_y_test_clusters(X_train_clusters, X_test_clusters, y_train_clusters, lr, max_iter, layers)
	y_pred = combine_preds(y_test_clusters, new_labels)
	return y_pred