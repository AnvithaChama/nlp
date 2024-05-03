# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the dataset
file_path = "Suicide_Detection.csv"  # Make sure to upload this file in the same directory if using an online IDE
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(data.head())
print("\n")

# Check for any null values and remove them
data.dropna(inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a Count Vectorizer to convert a collection of text documents to a vector of term/token counts
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

# Fit the vectorizer to the training data and transform the text
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Using Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test_vec)

# Calculate accuracy, precision, recall, and F1-score
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

# Function to predict text
def predict_text(text):
    text_vec = vectorizer.transform([text])
    pred = clf.predict(text_vec)
    return "Suicide tendency" if pred[0] == 1 else "No suicide tendency"

# Example usage
example_text1 = "I am very happy today!"
example_text2 = "I feel hopeless and think about giving up."

print("Predictions:")
print(f"Text: {example_text1}")
print(f"Prediction: {predict_text(example_text1)}")
print(f"Text: {example_text2}")
print(f"Prediction: {predict_text(example_text2)}")
