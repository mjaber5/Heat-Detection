import pandas as pd
from .clean_text import clean_tweet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['clean_tweet'] = df['tweet'].astype(str).apply(clean_tweet)
    return df

def load_and_prepare_data(filepath, test_size=0.2, random_state=42):
    df = load_and_clean_data(filepath)
    X = df['clean_tweet']
    y = df['class']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
