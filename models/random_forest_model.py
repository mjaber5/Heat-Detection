from sklearn.ensemble import RandomForestClassifier
from utils.evaluate import evaluate_model

def train_random_forest(X_train_vec, X_test_vec, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return evaluate_model(y_test, y_pred)