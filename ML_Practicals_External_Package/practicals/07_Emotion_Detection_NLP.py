# Install once if needed:
# pip install pandas numpy matplotlib scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("datasets")
if not DATA.exists():
    DATA = Path("../datasets")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(DATA / "emotion_text_demo.csv")
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size=0.25, random_state=42, stratify=df["emotion"])
model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), MultinomialNB())
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Emotion model accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\\nClassification Report:\\n", classification_report(y_test, pred, zero_division=0))
samples = ["I am so happy and excited", "I feel lonely and sad", "This makes me furious", "I am scared about the exam"]
for sentence, emotion in zip(samples, model.predict(samples)):
    print(f"{sentence} --> {emotion}")
