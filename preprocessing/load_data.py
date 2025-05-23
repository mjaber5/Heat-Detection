import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources (run once)
nltk.download('stopwords')

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_tweet(tweet):
    """Clean and preprocess a tweet."""
    tweet = tweet.lower()  # Convert to lowercase
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'\d+', '', tweet)  # Remove numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    # Remove stop words and apply stemming
    tweet = ' '.join([stemmer.stem(word) for word in tweet.split() if word not in stop_words])
    return tweet

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['tweet'] = df['tweet'].apply(clean_tweet)  # Assuming clean_tweet is defined
    X = df['tweet']
    y = df['class']
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8)
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     file_path = "path_to_your_dataset.csv"
#     X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data(file_path)
#     print(X_train.shape, X_test.shape)
