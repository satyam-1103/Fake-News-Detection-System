import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/cleaned_dataset1.csv")

def find_average_length_title_text(data):

    data["title_length"] = data["title"].apply(len)
    data["text_length"] = data["text"].apply(len)

    real_news = data[data["label"] == "Real"]
    fake_news = data[data["label"] == "Fake"]

    avg_real_title_len = real_news["title_length"].mean()
    avg_fake_title_len = fake_news["title_length"].mean()
    avg_real_text_len = real_news["text_length"].mean()
    avg_fake_text_len = fake_news["text_length"].mean()

    avg_lengths = {
        "Real": {"title": avg_real_title_len, "text": avg_real_text_len},
        "Fake": {"title": avg_fake_title_len, "text": avg_fake_text_len}
    }

    return avg_lengths

def print_avg_length(avg_lengths):
    print("Average Lengths:")
    for label, lengths in avg_lengths.items():
        print(f"{label} News:")
        print(f"- Average Title length: {lengths['title']:.2f} characters.")
        print(f"- Average Text length: {lengths['text']:.2f} characters.")

avg_lengths = find_average_length_title_text(data)
print_avg_length(avg_lengths)

def plot_average_lengths(avg_real_title_len, avg_fake_title_len, avg_real_text_len, avg_fake_text_len ):
    labels = ["Real Title", "Fake Title", "Real Text", "Fake Text"]
    lengths = [avg_real_title_len, avg_fake_title_len, avg_real_text_len, avg_fake_text_len]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, lengths, color=["green", "red", "green", "red"])
    plt.title("Average Title & Text Lengths of Real and Fake News")
    plt.ylabel("Average Length (characters)")
    plt.xticks(rotation=15)
    plt.show()

avg_real_title_len = avg_lengths["Real"]["title"]
avg_fake_title_len = avg_lengths["Fake"]["title"]
avg_real_text_len = avg_lengths["Real"]["text"]
avg_fake_text_len = avg_lengths["Fake"]["text"]

plot_average_lengths(avg_real_title_len, avg_fake_title_len, avg_real_text_len, avg_fake_text_len)




