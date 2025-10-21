# Hate Speech and Offensive Language Classification

A **Python-based text classification project** to detect hate speech, offensive language, and neutral content from tweets using classical machine learning models: **Logistic Regression** and **Support Vector Machine (SVM)** with TF-IDF features.

> ⚠️ **Disclaimer:** This repository is for **educational and research purposes only**. The dataset may contain offensive language. Do **not use this for production moderation** without proper evaluation.

---

## 📂 Dataset

The dataset is a public dataset (Davidson et al.) and contains the following columns:

| Column               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `count`              | Number of occurrences of this tweet                                         |
| `hate_speech`        | 1 if tweet contains hate speech, 0 otherwise                                |
| `offensive_language` | 1 if tweet is offensive but not hate speech                                  |
| `neither`            | 1 if tweet is neutral / neither offensive nor hate speech                   |
| `class`              | Target label: 0 = hate speech, 1 = offensive, 2 = neither                  |
| `tweet`              | Raw text of the tweet                                                       |

---

## ⚡ Features

- **Input:** raw tweet text (`tweet`)  
- **Output:** classification label (`class`) → `hate_speech`, `offensive`, or `neither`  
- **Preprocessing:**  
  - Lowercasing  
  - Removal of mentions (`@user`) and URLs  
  - Removal of special characters / punctuation  

---

## 🛠 Implementation

### 1. TF-IDF Vectorization
- Converts text to **numerical vectors**.  
- Uses **word n-grams (1-3)** and **stopwords removal**.  
- `max_features` set to 5000 for manageable dimensionality.

### 2. Models
- **Logistic Regression** with `class_weight='balanced'` → handles class imbalance.  
- **Support Vector Machine (LinearSVC)** with `class_weight='balanced'` → robust for high-dimensional text data.  

### 3. Evaluation
- Accuracy  
- Precision, Recall, F1-score per class  
- Confusion Matrix  
- Sample predictions for new tweets  

---

## 📈 Results (Sample Output)

**Logistic Regression:**
- Accuracy: 0.8492
- Class 0 (hate_speech) F1: 0.42, recall: 0.63
- Class 1 (offensive) F1: 0.90, recall: 0.85
- Class 2 (neither) F1: 0.84, recall: 0.94

**Support Vector Machine (SVM):**
- Accuracy: 0.8799
- Class 0 (hate_speech) F1: 0.41, recall: 0.44
- Class 1 (offensive) F1: 0.93, recall: 0.91
- Class 2 (neither) F1: 0.84, recall: 0.89

**Sample Predictions:**
- 'I hate you' -> hate_speech
- 'You're amazing!' -> neither
- 'Go away, loser' -> neither