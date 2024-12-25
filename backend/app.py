from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

app = Flask(__name__)
CORS(app)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')  
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')  

# Pre-load data and models
data = pd.read_excel("fake_news_data.xlsx")

def preprocess_text(text, stopwords_list, lemmatizer):
    """Preprocess the input text (cleaning, tokenizing, and lemmatizing)."""
    text_clean = re.sub(r"^[^-]*-\s*", "", text).lower()
    text_clean = re.sub(r"([^\w\s])", "", text_clean)
    text_clean = ' '.join([word for word in text_clean.split() if word not in stopwords_list])
    text_clean = [lemmatizer.lemmatize(token) for token in word_tokenize(text_clean)]
    return ' '.join(text_clean)

stopwords_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Combine title and text, then preprocess
data['combined_text'] = data['title'] + " " + data['text']
data['text_clean'] = data['combined_text'].apply(lambda x: preprocess_text(x, stopwords_list, lemmatizer))
X = data['text_clean']
Y = data['fake_or_factual']

# Vectorize the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Hyperparameter tuning for Logistic Regression
param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
grid.fit(X_vectorized, Y)
model = grid.best_estimator_

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, Y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Model Evaluation:")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Perform topic modeling with LDA
fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].apply(lambda x: x.split()).reset_index(drop=True)
dictionary_fake = corpora.Dictionary(fake_news_text)
doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

# Create LDA model
num_topics = 5
lda_model_fake = gensim.models.LdaModel(corpus=doc_term_fake,
                                         id2word=dictionary_fake,
                                         num_topics=num_topics,
                                         random_state=42)

# Print topics
topics = lda_model_fake.print_topics(num_words=10)
print("LDA Topics:")
for topic in topics:
    print(topic)

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to classify news articles."""
    input_data = request.json
    if not input_data or "text" not in input_data:
        return jsonify({"error": "Invalid input, please provide 'text' in JSON body."}), 400

    input_text = input_data["text"]
    processed_text = preprocess_text(input_text, stopwords_list, lemmatizer)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]

    return jsonify({"prediction": prediction})

@app.route("/topics", methods=["GET"])
def topics_endpoint():
    """API endpoint to retrieve LDA topics."""
    return jsonify({"topics": topics})

@app.route("/test", methods=["GET"])
def test():
    """Test endpoint to verify the API is running."""
    return jsonify({"message": "API is up and running!"})

if __name__ == "__main__":
    app.run(debug=True)
