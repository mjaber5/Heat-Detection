from sklearn.tree import DecisionTreeClassifier
from utils.evaluate import evaluate_model

def train_decision_tree_entropy(X_train_vec, X_test_vec, y_train, y_test):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return evaluate_model(y_test, y_pred)