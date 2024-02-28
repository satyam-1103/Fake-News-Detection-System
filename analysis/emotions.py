import re
import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

data = pd.read_csv("../dataset/cleaned_dataset1.csv")


nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

def detect_sentiment(data):

    sentiment_score = analyzer.polarity_scores(data)
    if sentiment_score["compound"] >= 0.05:
        return "Positive"
    elif sentiment_score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

data["Sentiment"] = data["text"].apply(detect_sentiment)

print(data[['text', 'Sentiment']].head())