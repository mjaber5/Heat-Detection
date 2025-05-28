from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
from utils.evaluate import evaluate_model

def train_decision_tree_gini(X_train_vec, X_test_vec, y_train, y_test, class_weight=None):
    model = DecisionTreeClassifier(criterion='gini', class_weight=class_weight, max_depth=10, min_samples_split=5, min_samples_leaf=2)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Visualize the decision tree
    #visualize_decision_tree(model, X_train_vec)
    
    metrics = evaluate_model(y_test, y_pred)
    return metrics, y_pred

# def visualize_decision_tree(model, X_train_vec):
#     # Create visualizations directory if it doesn't exist
#     os.makedirs('visualizations', exist_ok=True)
    
#     # Plot the decision tree
#     plt.figure(figsize=(20, 10))
#     plot_tree(model, 
#               feature_names=[f"feature_{i}" for i in range(X_train_vec.shape[1])], 
#               class_names=['Positive', 'Negative'],  # Updated for binary classification
#               filled=True, 
#               rounded=True, 
#               max_depth=3,  # Limit depth for readability
#               fontsize=10)
#     plt.title("Decision Tree (Gini) Visualization (Max Depth = 3)")
#     plt.savefig('visualizations/decision_tree_gini.png', dpi=300, bbox_inches='tight')
