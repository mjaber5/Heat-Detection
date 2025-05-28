from sklearn.neural_network import MLPClassifier
from utils.evaluate import evaluate_model

def train_neural_net(X_train_vec, X_test_vec, y_train, y_test, class_weight=None):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, learning_rate_init=0.001, 
                         random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred