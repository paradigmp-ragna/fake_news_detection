from sklearn.neural_network import MLPClassifier

def train_ann(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    model.fit(X_train, y_train)
    return model
