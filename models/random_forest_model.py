from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.evaluate import evaluate_model

def train_random_forest(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    return evaluate_model(y_test, y_pred)
