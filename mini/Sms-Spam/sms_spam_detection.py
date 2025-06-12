import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_recall_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import pickle
import json
from time import time
import multiprocessing
import sys
from joblib import parallel_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sms_spam.log'),
        logging.StreamHandler()
    ]
)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SpamDetector:
    def __init__(self):
        self.config = {
            'tfidf': {
                'max_features': 3000,  # Optimized for speed
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95,
                'stop_words': 'english'
            },
            'models': {
                'nb': {
                    'model': MultinomialNB(),
                    'params': {'model__alpha': [0.5, 1.0]}  # Reduced options
                },
                'svm': {
                    'model': SVC(probability=True, random_state=42),
                    'params': {'model__C': [1, 10], 'model__kernel': ['linear']}  # Simplified
                },
                'rf': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {'model__n_estimators': [100], 'model__max_depth': [10, 20]}
                }
            },
            'test_messages': [
                "Congratulations! You've won a $1000 gift card. Click here to claim.",
                "Hey, are we meeting at 6 PM today?",
                "URGENT: Your account has been compromised. Verify now!"
            ]
        }
        self.model = None
        self.vectorizer = None
        self.metrics = None

    def preprocess_text(self, text):
        """Enhanced text preprocessing with spam pattern highlighting"""
        if not isinstance(text, str):
            return ''
        
        text = text.lower()
        # Remove URLs/emails/phones
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
        # Highlight spam patterns
        spam_terms = r'\b(?:win|free|prize|claim|won|reward|urgent|click|verify)\b'
        text = re.sub(spam_terms, 'spamkeyword', text)
        # Clean special chars
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        return ' '.join(
            lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in stop_words and len(word) > 2
        )

    def load_data(self, filepath):
        """Load and prepare dataset"""
        try:
            data = pd.read_csv(filepath, encoding='latin-1')
            data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
            data['label'] = data['label'].map({'ham': 0, 'spam': 1})
            data['processed_message'] = data['message'].apply(self.preprocess_text)
            return data
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            sys.exit(1)

    def train_model(self, X_train, y_train, model_type='svm'):
        """Optimized model training with parallel processing"""
        with parallel_backend('threading', n_jobs=2):  # Safer multiprocessing
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('model', self.config['models'][model_type]['model'])
            ])
            
            grid_search = GridSearchCV(
                pipeline,
                self.config['models'][model_type]['params'],
                cv=5,
                scoring='f1',
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('precision_recall_curve.png')
        plt.close()
        
        return metrics

    def save_artifacts(self):
        """Save all model artifacts"""
        artifacts = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'metrics': self.metrics
        }
        
        with open('model_artifacts.pkl', 'wb') as f:
            pickle.dump(artifacts, f)
        
        with open('metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        with open('classification_report.txt', 'w') as f:
            f.write(self.metrics['classification_report'])

    def predict(self, message):
        """Make prediction with confidence score"""
        processed = self.preprocess_text(message)
        vectorized = self.vectorizer.transform([processed])
        proba = self.model.predict_proba(vectorized)[0][1]
        prediction = self.model.predict(vectorized)[0]
        return {
            'message': message,
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'confidence': float(proba if prediction == 1 else 1 - proba)
        }

    def run(self, args):
        try:
            start_time = time()
            
            # Load data
            data = self.load_data(args.dataset)
            logging.info(f"Loaded dataset with {len(data)} messages")

            # Feature extraction
            self.vectorizer = TfidfVectorizer(**self.config['tfidf'])
            X = self.vectorizer.fit_transform(data['processed_message'])
            y = data['label']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Train model
            logging.info(f"Training {args.model.upper()} model...")
            self.model, best_params = self.train_model(X_train, y_train, args.model)
            logging.info(f"Best parameters: {best_params}")

            # Evaluate
            self.metrics = self.evaluate_model(self.model, X_test, y_test)
            logging.info("\nClassification Report:\n" + self.metrics['classification_report'])
            logging.info(f"ROC AUC Score: {self.metrics['roc_auc']:.4f}")

            # Save artifacts
            self.save_artifacts()
            logging.info("Saved model artifacts")

            # Make predictions
            test_messages = [args.test] if args.test else self.config['test_messages']
            predictions = [self.predict(msg) for msg in test_messages]
            
            with open('predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
            
            for pred in predictions:
                logging.info(f"\nMessage: {pred['message']}")
                logging.info(f"Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.2%})")

            logging.info(f"Total execution time: {time() - start_time:.2f} seconds")
        
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="spam.csv", help="Path to dataset CSV")
    parser.add_argument("-t", "--test", default=None, help="Test message(s) to classify")
    parser.add_argument("-m", "--model", choices=['nb', 'svm', 'rf'], default='svm',
                      help="Model type: nb (Naive Bayes), svm (SVM), rf (Random Forest)")
    args = parser.parse_args()

    detector = SpamDetector()
    detector.run(args)
