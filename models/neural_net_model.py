from sklearn.neural_network import MLPClassifier
from utils.evaluate import evaluate_model

def train_neural_net(X_train_vec, X_test_vec, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train_vec, y_train)
    y_pred = mlp.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred