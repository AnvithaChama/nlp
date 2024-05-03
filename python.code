import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

file_path = "Suicide_Detection.csv"  # Make sure to upload this file in the same directory if using an online IDE
data = pd.read_csv(file_path)

print("First few rows of the dataset:")
print(data.head())
print("\n")

data.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='binary')
recall = metrics.recall_score(y_test, y_pred, average='binary')
f1_score = metrics.f1_score(y_test, y_pred, average='binary')

print(f"Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("\n")

def predict_text(text):
    text_vec = vectorizer.transform([text])
    pred = clf.predict(text_vec)
    return "Suicide tendency" if pred[0] == 1 else "No suicide tendency"

example_text1 = "I am very happy today!"
example_text2 = "I feel hopeless and think about giving up."

print("Predictions:")
print(f"Text: {example_text1}")
print(f"Prediction: {predict_text(example_text1)}")
print(f"Text: {example_text2}")
print(f"Prediction: {predict_text(example_text2)}")
