from preprocessing.load_data import load_and_prepare_data
from models.knn_model import train_knn
from models.svm_model import train_svm
from models.decision_tree_entropy import train_decision_tree_entropy
from models.decision_tree_gini import train_decision_tree_gini
from models.random_forest_model import train_random_forest
from models.neural_net_model import train_neural_net
from utils.exploare_data import plot_confusion_matrix

if __name__ == '__main__':
    # Load and split data, unpacking six values including class weights
    X_train_vec, X_test_vec, y_train, y_test, vectorizer, class_weights = load_and_prepare_data('data/labeled_data.csv')
    
    # Dictionary to map model functions to names for visualization
    models = {
        'K-Nearest Neighbors (KNN)': lambda x_train, x_test, y_train, y_test: train_knn(x_train, x_test, y_train, y_test),
        'Support Vector Machine (SVM)': lambda x_train, x_test, y_train, y_test: train_svm(x_train, x_test, y_train, y_test, class_weights),
        'Decision Tree (Entropy)': lambda x_train, x_test, y_train, y_test: train_decision_tree_entropy(x_train, x_test, y_train, y_test, class_weights),
        'Decision Tree (Gini)': lambda x_train, x_test, y_train, y_test: train_decision_tree_gini(x_train, x_test, y_train, y_test, class_weights),
        'Random Forest': lambda x_train, x_test, y_train, y_test: train_random_forest(x_train, x_test, y_train, y_test, class_weights),
        'Neural Network': lambda x_train, x_test, y_train, y_test: train_neural_net(x_train, x_test, y_train, y_test, class_weights)
    }

    # Train and evaluate each model
    for model_name, train_func in models.items():
        print(f"\n--- {model_name} ---")
        # Train the model and get metrics and predictions
        metrics, y_pred = train_func(X_train_vec, X_test_vec, y_train, y_test)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name)