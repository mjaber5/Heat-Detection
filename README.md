# 🐦 Tweet Classification with Machine Learning

This project focuses on classifying tweets using various supervised machine learning algorithms. It leverages natural language processing (NLP) techniques like TF-IDF vectorization, and implements several models to analyze and compare performance.

## 🚀 Features

- Preprocessing and cleaning of tweet data.
- Feature extraction using TF-IDF.
- Implementation of multiple ML algorithms:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree (Gini & Entropy)
  - Random Forest
  - Neural Network (MLPClassifier)
- Evaluation using standard metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## 🗂️ Project Structure

```
tweet_classification/
│
├── data/                          # CSV dataset files
│   └── labeled_data.csv
│
├── models/                        # Contains each model's training function
│   ├── knn_model.py
│   ├── svm_model.py
│   ├── decision_tree_entropy.py
│   ├── decision_tree_gini.py
│   ├── random_forest_model.py
│   └── neural_net_model.py
│
├── preprocessing/                # Data loading and cleaning logic
│   └── load_data.py
│
├── utils/                         # Utility scripts
│   └── evaluate.py               # Evaluation metrics
│
├── main.py                        # Entry point script for training & evaluation
└── README.md                      # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/tweet_classification.git
   cd tweet_classification
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

Run the main script:

```bash
python main.py
```

Output will include evaluation metrics (Accuracy, Precision, Recall, F1 Score) for each model.

Ensure that the dataset (`labeled_data.csv`) is available inside the `data/` directory and has at least these columns:
- `clean_tweet` — cleaned text content
- `class` — target label for classification

## 🧪 Evaluation Metrics

Each model is evaluated using:

- `Accuracy`: Overall correctness
- `Precision`: Quality of positive predictions
- `Recall`: Coverage of actual positive instances
- `F1 Score`: Balance between precision and recall

Results are printed after training each model.

## ✅ Requirements

- Python 3.8+
- pandas
- scikit-learn

If not available, install manually:

```bash
pip install pandas scikit-learn
```

Or use the provided `requirements.txt`.

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss.

---

👤 Created by [Mohammed Jaber](https://github.com/mjaber5) – connect on [LinkedIn](https://www.linkedin.com/in/mohammad-jaber-profile?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
