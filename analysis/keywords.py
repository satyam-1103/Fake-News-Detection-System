import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd

from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')


data = pd.read_csv("../dataset/cleaned_dataset1.csv")

def detect_keywords(data):
    stop_words = set(stopwords.words("english"))
    title_counter = Counter()
    text_counter = Counter()

    for index, row in data.iterrows():
        title_words = word_tokenize(row["title"])
        text_words = word_tokenize(row["text"])

        title_words = [word.lower() for word in title_words if word.isalpha() and word.lower() not in stop_words]
        text_words = [word.lower() for word in text_words if word.isalpha() and word.lower() not in stop_words]

        if row["label"] == "Fake":
            title_counter.update(title_words)
            text_counter.update(text_words)

    top_keywords_title = title_counter.most_common(10)
    top_keywords_text = text_counter.most_common(10)

    return top_keywords_title, top_keywords_text

top_keywords_title, top_keywords_text = detect_keywords(data)
print("Top 10 Keywords Associated with Fake News Title:")
for keyword, count in top_keywords_title:
    print(f'{keyword} : {count}')

print("\nTop 10 Keywords Associated with Fake News Text:")
for keyword, count in top_keywords_text:
    print(f'{keyword} : {count}')
