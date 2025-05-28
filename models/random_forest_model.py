from sklearn.ensemble import RandomForestClassifier
from utils.evaluate import evaluate_model

def train_random_forest(X_train_vec, X_test_vec, y_train, y_test, class_weight=None):
    model = RandomForestClassifier(n_estimators=100, class_weight=class_weight, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred