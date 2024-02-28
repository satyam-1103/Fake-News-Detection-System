import re
import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd
from scipy.stats import chi2_contingency

data = pd.read_csv("../dataset/cleaned_dataset1.csv")

def detect_sensationalism(data):
    sensational_keywords = [
        "Shocking", "Explosive", "Sensational", "Bombshell", "Outrageous",
        "Scandalous", "Terrifying", "Disturbing", "Unbelievable", "Exposed",
        "Revealed", "Breathtaking", "Jaw-dropping", "Mind-blowing",
        "Apocalyptic", "Catastrophic", "Nightmare", "Horrifying", "Insane",
        "Massive", "Incredible", "Epic", "Dramatic", "Intense",
        "Gripping", "Inflammatory", "Frightening", "Alarming", "Tremendous",
        "Hair-raising", "Heart-stopping", "Blood-curdling", "Bone-chilling",
        "Eye-popping", "Mind-boggling", "Astonishing", "Astounding", "Extraordinary",
        "Intriguing", "Startling", "Unreal", "Staggering", "Phenomenal",
        "Overwhelming", "Unprecedented", "Revolutionary", "Groundbreaking",
        "Unthinkable", "Monumental", "Unimaginable", "Unforgettable", "Eerie",
        "Spine-tingling", "Spectacular", "Fantastic", "Marvelous", "Wondrous",
        "Mysterious", "Enigmatic", "Puzzling", "Bizarre", "Curious",
        "Provocative", "Compelling", "Harrowing", "Exhilarating", "Electrifying",
        "Astounding", "Stunning", "Remarkable", "Incredible", "Unbelievable",
        "Awesome", "Awe-inspiring", "Eye-opening", "Heart-pounding", "Gut-wrenching",
        "Emotionally-charged", "Heartbreaking", "Soul-stirring", "Spellbinding", "Riveting",
        "Mind-expanding", "Life-changing", "Game-changing", "Paradigm-shifting", "Unfathomable",
        "Incomprehensible", "Unsettling", "Mind-altering", "Life-altering", "Life-threatening",
        "Chilling", "Menacing", "Horrifying", "Spine-chilling", "Gruesome",
        "Fatal", "Lethal", "Deadly", "Disastrous", "Apocalyptic"
    ]

    for keyword in sensational_keywords:
        if re.search(r'\b' + keyword + r'\b', data, re.IGNORECASE):
            return True
        return False

data["Sensationalism"] = data["text"].apply(detect_sensationalism)
contigency_table = pd.crosstab(data["Sensationalism"], data["label"])
print(contigency_table)


chi2, p, _, _ = chi2_contingency(contigency_table)

print(f"Chi-squared statistics: {chi2}")
print(f"p-value: {p}")

alpha = 0.05

if p < alpha:
    print("There is a significant association between sensationalism and credibility")
else:
    print("There is not significant association between sensationalism and credibility")