import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import os

# Download NLTK resources if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# --- Custom Transformers for Pipeline ---

class SentimentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is a Series of text
        sentiments = X.apply(lambda x: self.sia.polarity_scores(str(x))['compound'])
        return sentiments.values.reshape(-1, 1)

class POSRatioExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is a Series of text
        def get_pos_ratios(text):
            text = str(text)
            tokens = nltk.word_tokenize(text)
            if not tokens:
                return [0, 0, 0]
            tags = nltk.pos_tag(tokens)
            total = len(tokens)
            noun_count = sum(1 for _, t in tags if t.startswith('NN'))
            verb_count = sum(1 for _, t in tags if t.startswith('VB'))
            pron_count = sum(1 for _, t in tags if t in ['PRP', 'PRP$', 'WP', 'WP$'])
            return [noun_count/total, verb_count/total, pron_count/total]
        
        ratios = X.apply(get_pos_ratios)
        return np.array(ratios.tolist())

# --- Main Training Flow ---

def train_and_save_model():
    print("Loading data...")
    dataset_path = "Politifact_Dataset/politifact_factcheck_data.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    
    # Filter for relevant sources
    allowed_sources = ['news', 'blog', 'social_media']
    df = df[df['statement_source'].str.lower().isin(allowed_sources)].reset_index(drop=True)
    
    # Label Encoding (True/Mostly-True -> Real, else -> Fake)
    # The notebook logic used: 'true' in verdict lower -> real, else fake.
    # We will stick to the notebook logic for consistency with the 74% accuracy result.
    df['label'] = df['verdict'].apply(lambda x: 1 if 'true' in str(x).lower() else 0)
    
    X = df['statement'].astype(str)
    y = df['label']
    
    print(f"Training on {len(df)} samples...")
    
    # Define Pipeline
    # 1. Text Features (TF-IDF)
    # 2. Auxiliary Features (Sentiment, POS) - Scaled
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('auxiliary', Pipeline([
                ('features', FeatureUnion([
                    ('sentiment', SentimentExtractor()),
                    ('pos', POSRatioExtractor())
                ])),
                ('scaler', StandardScaler())
            ]))
        ])),
        ('classifier', LogisticRegression(max_iter=1000, C=0.6, solver='liblinear')) # Params from notebook tuning
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save
    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved to model_pipeline.pkl")

if __name__ == "__main__":
    train_and_save_model()
