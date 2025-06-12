#!/usr/bin/env python
# extract_model.py - Extracts model and vectorizer from model_artifacts.pkl

import pickle
import os
import sys

def extract_models():
    print("Extracting model components from model_artifacts.pkl...")
    
    # Check if model_artifacts.pkl exists
    if not os.path.exists('model_artifacts.pkl'):
        print("Error: model_artifacts.pkl not found")
        print("Please run FINAL_SMS_SPAM_DETECTION.py first")
        return False
    
    try:
        # Load the combined model
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        # Extract components
        model = artifacts.get('model')
        vectorizer = artifacts.get('vectorizer')
        
        if model is None:
            print("Error: Model not found in artifacts")
            return False
            
        if vectorizer is None:
            print("Error: Vectorizer not found in artifacts")
            return False
        
        # Save as separate files
        print("Saving model.pkl...")
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        print("Saving vectorizer.pkl...")
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
            
        print("Success! Extracted model and vectorizer.")
        print("You can now run app.py")
        return True
        
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return False

if __name__ == "__main__":
    success = extract_models()
    sys.exit(0 if success else 1)