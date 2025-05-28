import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Download NLTK resources (run once)
nltk.download('stopwords')

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
# Add domain-specific stopwords (e.g., common Twitter terms or slang)
custom_stopwords = {'rt', 'amp', 'lol', 'lmao', 'smh', 'tho', 'ya', 'yall', 'u', 'ur', 'nig', 'nigga', 'niggah'}
stop_words.update(custom_stopwords)
stemmer = PorterStemmer()

def clean_tweet(tweet):
    """Clean and preprocess a tweet."""
    if not isinstance(tweet, str):  # Handle non-string inputs
        return ''
    
    tweet = tweet.lower()  # Convert to lowercase
    
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    tweet = re.sub(r'@\w+|#\w+', '', tweet)
    
    # Remove emojis (basic Unicode range for emojis)
    tweet = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', tweet)
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    
    # Remove punctuation and special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove extra whitespace
    tweet = ' '.join(tweet.split())
    
    # Remove stop words and apply stemming
    tweet = ' '.join([stemmer.stem(word) for word in tweet.split() if word not in stop_words and len(word) > 2])
    
    return tweet

def load_and_prepare_data(file_path):
    """Load and preprocess dataset for binary classification."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Clean tweets
    df['tweet'] = df['tweet'].apply(clean_tweet)
    
    # Convert to binary classification: 0 (Hate Speech) and 1 (Offensive Language) -> 1 (Negative), 2 (Neither) -> 0 (Positive)
    df['binary_class'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)
    
    # Remove empty tweets after cleaning
    df = df[df['tweet'].str.strip() != '']
    
    # Features and labels
    X = df['tweet']
    y = df['binary_class']
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased to capture more features
        ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.85,  # Ignore terms that appear in more than 85% of documents
        sublinear_tf=True  # Apply sublinear term frequency scaling
    )
    X_vec = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    return X_train, X_test, y_train, y_test, vectorizer, class_weight_dict

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     file_path = "data/labeled_data.csv"
#     X_train, X_test, y_train, y_test, vectorizer, class_weights = load_and_prepare_data(file_path)
#     print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     print(f"Class distribution: {np.bincount(y_train)}")
#     print(f"Class weights: {class_weights}")