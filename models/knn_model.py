from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from utils.evaluate import evaluate_model

def train_knn(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return evaluate_model(y_test, y_pred)
