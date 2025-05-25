import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, model_name):
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Hate Speech', 'Offensive Language', 'Neither'],
                yticklabels=['Hate Speech', 'Offensive Language', 'Neither'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'visualizations/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()