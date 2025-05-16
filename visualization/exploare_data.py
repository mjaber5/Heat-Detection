import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df, label_column='class'):
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[label_column])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
