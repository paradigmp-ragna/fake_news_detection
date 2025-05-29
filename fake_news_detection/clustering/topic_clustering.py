from sklearn.cluster import KMeans

def perform_clustering(X, n_clusters=2):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model.labels_
