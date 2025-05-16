from preprocessing.load_data import load_and_prepare_data
from models.knn_model import train_knn
from models.svm_model import train_svm
from models.decision_tree_entropy import train_decision_tree_entropy
from models.decision_tree_gini import train_decision_tree_gini
from models.random_forest_model import train_random_forest
from models.neural_net_model import train_neural_net

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data('data/labeled_data.csv')

    print("\n--- K-Nearest Neighbors (KNN) ---")
    metrics = train_knn(X_train, X_test, y_train, y_test)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Similarly for other models
