# NLP Disaster Classification

This repository contains a Python code for classifying disaster-related text messages using Natural Language Processing (NLP) techniques and different classifiers. The goal is to predict whether a given text message represents a real disaster (target=1) or not (target=0).

## Dataset
The dataset used for this task can be found on Kaggle: [NLP Getting Started - Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started). It contains labeled disaster-related text messages along with their binary target labels (1 for real disaster, 0 for not a real disaster).

## Requirements
Make sure to install the required libraries before running the code. You can install the necessary libraries using the following command:

```bash
pip install pandas numpy scikit-learn spacy
python -m spacy download en_core_web_sm
```

## Code Overview
1. Data Preprocessing: The text data is preprocessed by tokenizing and lemmatizing the text using the spaCy English language model.

2. Model Selection and Hyperparameter Tuning: The code uses the HashingVectorizer for text vectorization and allows you to experiment with different classifiers, including Support Vector Machine (SVM), Multinomial Naive Bayes, and Random Forest. The code performs hyperparameter tuning using RandomizedSearchCV from scikit-learn.

3. Evaluation: The code evaluates the classifier's performance using accuracy on a validation set.

4. Submission: The code makes predictions on the test set and saves the results in a CSV file for submission.

## Usage
1. Ensure the `train.csv` and `test.csv` files are placed in the same directory as the script.

2. Run the script using the following command:

```bash
python nlp_disaster_classification.py
```

3. The script will print the validation accuracy and save the predictions in a file named `submission.csv` for submission to Kaggle.

Feel free to explore and modify the code to experiment with different classifiers and NLP techniques for better accuracy. Happy coding and best of luck with your NLP journey! 🚀
