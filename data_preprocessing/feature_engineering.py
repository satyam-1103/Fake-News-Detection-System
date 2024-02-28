import re
import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("../dataset/cleaned_dataset1.csv")

def find_top_keywords_fake_news(data):
    # Filter fake news data
    fake_news_data = data[data['label'] == "Fake"]

    # Vectorize text data
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(fake_news_data["text"])

    # Calculate word frequencies
    word_frequencies = X.toarray().sum(axis=0)

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Extract top keywords
    top_keywords_indices = word_frequencies.argsort()[-10:][::-1]
    keywords = [feature_names[i] for i in top_keywords_indices]

    return keywords

keywords = find_top_keywords_fake_news(data)
print("Top Keywords Associated with Fake News:")
print(keywords)