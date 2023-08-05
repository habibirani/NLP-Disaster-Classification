# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Loading the spaCy English language model
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import HashingVectorizer
# Load the dataset (assuming 'train.csv' and 'test.csv' are in the same directory)
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

# Data preprocessing
def tokenize_and_lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(tokens)

X_train = train_data['text'].apply(tokenize_and_lemmatize)
X_test = test_data['text'].apply(tokenize_and_lemmatize)
y_train = train_data['target']

# Create a Pipeline with HashingVectorizer and Classifier
pipeline = Pipeline([
    ('vectorizer', HashingVectorizer(n_features=2**18)),
    ('classifier', SVC())  # Change this to any classifier you want to try (e.g., MultinomialNB or RandomForestClassifier)
])

# Define the hyperparameter grid for the classifier
param_distributions = {
    'classifier__C': [1, 10, 100],
    'classifier__kernel': ['linear', 'rbf']
}

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, cv=3, n_iter=10, verbose=1)
random_search.fit(X_train, y_train)

# Get the best classifier from the RandomizedSearchCV
best_classifier = random_search.best_estimator_

# Make predictions on the validation set
y_pred = best_classifier.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Make predictions on the test set
y_test_pred = best_classifier.predict(X_test)

# Save the predictions to a CSV file for submission
submission = pd.DataFrame({'id': test_data['id'], 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
