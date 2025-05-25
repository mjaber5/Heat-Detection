from sklearn.neighbors import KNeighborsClassifier
from utils.evaluate import evaluate_model

def train_knn(X_train_vec, X_test_vec, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred