# Section 1: Importing Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Ensure NLTK downloads are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Section 2: Loading and Inspecting the Data
file_path = '' #Upload your train file path
data = pd.read_csv(file_path)

print(data.head())
print(data.info())
print(data.isnull().sum())

# Section 3: Data Preprocessing
def clean_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', str(text))
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words into a sentence
    return ' '.join(words)

data['comment_text'] = data['comment_text'].apply(clean_text)

# Section 4: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['comment_text'])
y = data['toxic']  # Ensure the column exists in your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 5: Building a Model
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))

# Section 6: Model Tuning
parameters = {'alpha': [0.1, 0.5, 1, 2, 5, 10]}
grid_search = GridSearchCV(MultinomialNB(), param_grid=parameters, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate on Test Set
y_test_pred = best_model.predict(X_test)
print('Tuned Accuracy:', accuracy_score(y_test, y_test_pred))
print('Tuned Precision:', precision_score(y_test, y_test_pred))
print('Tuned Recall:', recall_score(y_test, y_test_pred))
print('Tuned F1-score:', f1_score(y_test, y_test_pred))

# Predictions on Example Comments
example_comments = ['This is a nice comment.', 'I hate you!', 'You are stupid.', 'I love this product.']
example_cleaned = [clean_text(comment) for comment in example_comments]
example_vectors = vectorizer.transform(example_cleaned)
example_preds = best_model.predict(example_vectors)

for i, comment in enumerate(example_comments):
    print(f"Comment: {comment}")
    print(f"Toxic: {bool(example_preds[i])}")
