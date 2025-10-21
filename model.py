import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("hate_speech_dataset.csv")

def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

df['clean_text'] = df['tweet'].apply(clean_text)

X = df['clean_text']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,3),
    analyzer='word',
    stop_words='english'
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("--- Logistic Regression ---")

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

class_mapping = {0: "hate_speech", 1: "offensive", 2: "neither"}
samples = ["I hate you", "You're amazing!", "Go away, loser"]
predictions = model.predict(tfidf.transform(samples))
pred_labels = [class_mapping[p] for p in predictions]

for text, label in zip(samples, pred_labels):
    print(f"Text: '{text}' -> Predicted class: {label}")

print("--- Support Vector Machine ---")

svm_model = LinearSVC(class_weight='balanced', max_iter=5000)
svm_model.fit(X_train_tfidf, y_train)

y_pred = svm_model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

pred_labels = [class_mapping[p] for p in svm_model.predict(tfidf.transform(samples))]

for text, label in zip(samples, pred_labels):
    print(f"Text: '{text}' -> Predicted class: {label}")