from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train_neural_net(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics
