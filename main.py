from preprocessing.load_data import load_and_prepare_data
from models.knn_model import train_knn
from models.svm_model import train_svm
from models.decision_tree_entropy import train_decision_tree_entropy
from models.decision_tree_gini import train_decision_tree_gini
from models.random_forest_model import train_random_forest
from models.neural_net_model import train_neural_net

if __name__ == '__main__':
    # Load and split data, unpacking five values and ignoring the vectorizer
    X_train_vec, X_test_vec, y_train, y_test, _ = load_and_prepare_data('data/labeled_data.csv')
    
    # Train and evaluate each model using the pre-vectorized data
    print("\n--- K-Nearest Neighbors (KNN) ---")
    metrics = train_knn(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n--- Support Vector Machine (SVM) ---")
    metrics = train_svm(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n--- Decision Tree (Entropy) ---")
    metrics = train_decision_tree_entropy(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n--- Decision Tree (Gini) ---")
    metrics = train_decision_tree_gini(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n--- Random Forest ---")
    metrics = train_random_forest(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\n--- Neural Network ---")
    metrics = train_neural_net(X_train_vec, X_test_vec, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")