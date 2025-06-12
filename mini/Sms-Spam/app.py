from flask import Flask, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import deque
import time
import json
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create a deque to store recent messages (limit to 10)
recent_messages = deque(maxlen=10)

# Add this to track statistics
stats = {
    'total': 0,
    'spam': 0,
    'ham': 0
}

# Load models - first try individual files, then try model_artifacts.pkl
def load_models():
    try:
        # Try to load individual files first
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            logging.info("Loading model and vectorizer from individual files...")
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        
        # If individual files not found, try loading from artifacts
        elif os.path.exists('model_artifacts.pkl'):
            logging.info("Loading from combined model_artifacts.pkl...")
            with open('model_artifacts.pkl', 'rb') as f:
                artifacts = pickle.load(f)
                model = artifacts.get('model')
                vectorizer = artifacts.get('vectorizer')
                if model is None or vectorizer is None:
                    raise ValueError("Model or vectorizer not found in artifacts")
                return model, vectorizer
        else:
            raise FileNotFoundError("No model files found")
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise e

try:
    model, vectorizer = load_models()
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    logging.error("Make sure to run the training script first")
    model = None
    vectorizer = None

def preprocess_text(text):
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

def generate_explanation(message, prediction, confidence):
    """Generate a more detailed user-friendly explanation based on the prediction and message content"""
    confidence_value = float(confidence.strip('%')) / 100
    
    # Check for common spam indicators
    has_urls = bool(re.search(r'http\S+|www\S+|https\S+', message))
    has_spam_words = bool(re.search(r'\b(?:free|win|prize|claim|won|reward|urgent|click|verify|cash|money|exclusive|limited|offer|congrat|gift)\b', message.lower()))
    has_exclamation = '!' in message
    has_capitalization = len(re.findall(r'[A-Z]{2,}', message)) > 0
    message_length = len(message)
    
    # Classify confidence levels
    confidence_level = ""
    if confidence_value > 0.95:
        confidence_level = "very high"
    elif confidence_value > 0.85:
        confidence_level = "high"
    elif confidence_value > 0.75:
        confidence_level = "moderate"
    else:
        confidence_level = "low"
    
    if prediction == 'Spam':
        if confidence_value > 0.9:
            explanation = f"This message is highly likely to be spam ({confidence_level} confidence). "
            reasons = []
            if has_urls:
                reasons.append("contains suspicious links")
            if has_spam_words:
                reasons.append("includes common spam trigger words")
            if has_exclamation or has_capitalization:
                reasons.append("uses excessive punctuation or capitalization typical of spam")
                
            if reasons:
                explanation += "It " + ", ".join(reasons) + ". "
            explanation += "You should avoid clicking any links or responding to this message."
        else:
            explanation = f"This message appears to be spam ({confidence_level} confidence). "
            explanation += "It contains some characteristics of spam messages but not all typical indicators. "
            explanation += "Exercise caution before responding or clicking any links."
    else:  # Ham
        if confidence_value > 0.9:
            explanation = f"This message is highly likely to be legitimate ({confidence_level} confidence). "
            if message_length < 20:
                explanation += "It's a short, common message with no suspicious content. "
            else:
                explanation += "It doesn't contain suspicious links or common spam phrases. "
            explanation += "It appears safe for normal interaction."
        else:
            explanation = f"This message appears to be legitimate ({confidence_level} confidence). "
            explanation += "While it doesn't have strong spam indicators, exercise normal caution when responding."
    
    return explanation

@app.route('/')
def home():
    return '''
    <html>
      <head>
        <title>SMS Spam Detector</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
        <style>
          body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
            background-attachment: fixed;
            color: #333;
            min-height: 100vh;
          }
          .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
          }
          h1 { 
            color: #004d7a; 
            text-align: center;
            margin-bottom: 30px;
            font-weight: 600;
          }
          textarea { 
            width: 100%; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 8px;
            font-family: inherit;
            resize: vertical;
            box-sizing: border-box;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            font-size: 16px;
            transition: border 0.3s ease;
          }
          textarea:focus {
            outline: none;
            border-color: #004d7a;
            box-shadow: 0 0 0 2px rgba(0, 77, 122, 0.2);
          }
          button { 
            background: #004d7a; 
            color: white; 
            padding: 12px 20px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: background 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
          }
          button i {
            margin-right: 8px;
          }
          button:hover { 
            background: #003b5c; 
          }
          #result { 
            margin-top: 20px; 
            padding: 15px; 
            border-radius: 8px;
            font-size: 18px;
            font-weight: 500;
            transition: all 0.3s ease;
            display: none;
          }
          .explanation {
            margin-top: 15px;
            font-size: 16px;
            line-height: 1.5;
            color: #333;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #004d7a;
          }
          .spam { 
            background-color: #ffcccc; 
            border-left: 5px solid #ff4d4d;
            color: #cc0000;
          }
          .ham { 
            background-color: #ccffcc; 
            border-left: 5px solid #4CAF50;
            color: #1b5e20;
          }
          nav { 
            margin-bottom: 30px;
            display: flex;
            justify-content: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
          }
          nav a { 
            margin: 0 20px; 
            color: #004d7a; 
            text-decoration: none;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.3s ease;
          }
          nav a:hover {
            background: rgba(0, 77, 122, 0.1);
          }
          nav a.active {
            background: #004d7a;
            color: white;
          }
          .stats { 
            margin-top: 30px; 
            background: #f5f5f5; 
            padding: 20px; 
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
          }
          table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
          }
          th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
          }
          th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #004d7a;
          }
          tr:hover {
            background-color: #f8f9fa;
          }
          tr:last-child td {
            border-bottom: none;
          }
          .delete-btn {
            background: #ff4d4d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 10px;
            cursor: pointer;
            transition: background 0.3s ease;
          }
          .delete-btn:hover {
            background: #cc0000;
          }
          .chart-container {
            margin-top: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
          }
          .pie-chart {
            width: 200px;
            height: 200px;
            margin: 20px auto;
          }
          .action-btn {
            margin-top: 20px;
          }
          #empty-history {
            text-align: center;
            padding: 30px;
            color: #666;
          }
          .section-header {
            font-size: 20px;
            color: #004d7a;
            margin-bottom: 15px;
            font-weight: 600;
          }
          .timestamp {
            color: #666;
            font-size: 14px;
          }
          .confidence-indicator {
            display: inline-block;
            height: 8px;
            border-radius: 4px;
            min-width: 50px;
            margin-right: 10px;
            vertical-align: middle;
          }
          .spam-confidence {
            background: linear-gradient(90deg, #ffcccc 0%, #ff4d4d 100%);
          }
          .ham-confidence {
            background: linear-gradient(90deg, #ccffcc 0%, #4CAF50 100%);
          }
          .footer {
            text-align: center; 
            margin-top: 40px;
            color: rgba(255,255,255,0.7);
            font-size: 14px;
          }
          .loading {
            text-align: center;
            padding: 20px;
            color: #004d7a;
          }
          
          .loading i {
            margin-right: 10px;
          }
          
          .confidence-meter {
            margin-top: 15px;
          }
          
          .confidence-label {
            font-size: 14px;
            margin-bottom: 5px;
            color: #555;
          }
          
          .confidence-bar-container {
            height: 12px;
            background-color: #f0f0f0;
            border-radius: 6px;
            overflow: hidden;
          }
          
          .confidence-bar {
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
          }
          
          .prediction-header {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 18px;
          }
          
          .spam .prediction-header {
            color: #cc0000;
          }
          
          .ham .prediction-header {
            color: #1b5e20;
          }
          
          .spam .prediction-header i {
            color: #ff4d4d;
          }
          
          .ham .prediction-header i {
            color: #4CAF50;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1><i class="fas fa-shield-alt"></i> SMS Spam Detector</h1>
          <nav>
            <a href="#" id="home-link" class="active"><i class="fas fa-home"></i> Home</a>
            <a href="#" id="stats-link"><i class="fas fa-chart-pie"></i> Stats</a>
            <a href="#" id="history-link"><i class="fas fa-history"></i> History</a>
          </nav>
          
          <div id="home-section">
            <form id="form">
              <textarea id="message" rows="4" placeholder="Enter your SMS message here..."></textarea><br><br>
              <button type="submit"><i class="fas fa-search"></i> Check Message</button>
            </form>
            <div id="result"></div>
            <div id="explanation" class="explanation" style="display:none;"></div>
          </div>
          
          <div id="stats-section" class="stats" style="display:none;">
            <div class="section-header"><i class="fas fa-chart-pie"></i> Statistics</div>
            <div id="stats-content"></div>
            <div id="pie-chart" class="pie-chart"></div>
          </div>
          
          <div id="history-section" style="display:none;">
            <div class="section-header"><i class="fas fa-history"></i> Recent Messages</div>
            <div id="history-content"></div>
          </div>
        </div>
        
        <div class="footer">© 2025 SMS Spam Detector - Powered by ML</div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
        <script>
          // Navigation
          const sections = {
            'home-link': 'home-section',
            'stats-link': 'stats-section',
            'history-link': 'history-section'
          };
          
          Object.keys(sections).forEach(linkId => {
            document.getElementById(linkId).addEventListener('click', function(e) {
              e.preventDefault();
              
              // Hide all sections
              Object.values(sections).forEach(sectionId => {
                document.getElementById(sectionId).style.display = 'none';
              });
              
              // Remove active class from all links
              Object.keys(sections).forEach(id => {
                document.getElementById(id).classList.remove('active');
              });
              
              // Show the selected section
              document.getElementById(sections[linkId]).style.display = 'block';
              this.classList.add('active');
              
              // Load data if needed
              if (linkId === 'stats-link') loadStats();
              if (linkId === 'history-link') loadHistory();
            });
          });
          
          // Form submission
          document.getElementById('form').onsubmit = function(e) {
            e.preventDefault();
            const messageInput = document.getElementById('message');
            const resultDiv = document.getElementById('result');
            const explanationDiv = document.getElementById('explanation');
            
            if (!messageInput.value.trim()) {
              resultDiv.innerText = 'Please enter a message to check';
              resultDiv.className = '';
              resultDiv.style.display = 'block';
              explanationDiv.style.display = 'none';
              return;
            }
            
            // Show loading state
            resultDiv.innerHTML = `<div class="loading"><i class="fas fa-circle-notch fa-spin"></i> Analyzing message...</div>`;
            resultDiv.className = '';
            resultDiv.style.display = 'block';
            explanationDiv.style.display = 'none';
            
            fetch('/predict', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({message: messageInput.value})
            })
            .then(response => response.json())
            .then(data => {
              // Convert confidence to a number for calculations
              const confidenceNum = parseFloat(data.confidence) / 100 || 0;
              
              // Color gradient based on confidence
              let confidenceColor;
              if (data.prediction === 'Spam') {
                // Red gradient for spam (darker = more confident)
                const redIntensity = Math.min(255, Math.floor(180 + (confidenceNum * 75)));
                confidenceColor = `rgba(${redIntensity}, 70, 70, 1)`;
              } else {
                // Green gradient for ham (darker = more confident)
                const greenIntensity = Math.min(255, Math.floor(100 + (confidenceNum * 155)));
                confidenceColor = `rgba(70, ${greenIntensity}, 70, 1)`;
              }
              
              // Create a more visual confidence meter
              const confidenceMeter = `
                <div class="confidence-meter">
                  <div class="confidence-label">${data.confidence} confidence</div>
                  <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: ${confidenceNum * 100}%; background-color: ${confidenceColor};"></div>
                  </div>
                </div>
              `;
              
              // Display the prediction with the confidence meter
              resultDiv.innerHTML = `
                <div class="prediction-header">
                  <i class="${data.prediction === 'Spam' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle'}"></i>
                  <strong>${data.prediction === 'Spam' ? 'Spam Detected' : 'Message is Safe'}</strong>
                </div>
                ${confidenceMeter}
              `;
              
              resultDiv.className = data.prediction.toLowerCase();
              resultDiv.style.display = 'block';
              
              // Display explanation
              if (data.explanation) {
                explanationDiv.innerText = data.explanation;
                explanationDiv.style.display = 'block';
              } else {
                explanationDiv.style.display = 'none';
              }
              
              // Clear input after successful submission
              messageInput.value = '';
            })
            .catch(error => {
              resultDiv.innerText = 'Error: ' + error.message;
              resultDiv.className = '';
              resultDiv.style.display = 'block';
              explanationDiv.style.display = 'none';
            });
          };
          
          // Load stats
          function loadStats() {
            fetch('/stats')
              .then(response => response.json())
              .then(data => {
                const statsSection = document.getElementById('stats-content');
                statsSection.innerHTML = `
                  <table>
                    <tr>
                      <td><strong>Total messages analyzed:</strong></td>
                      <td>${data.total_messages}</td>
                    </tr>
                    <tr>
                      <td><strong>Spam messages:</strong></td>
                      <td>${data.spam_count}</td>
                    </tr>
                    <tr>
                      <td><strong>Ham messages:</strong></td>
                      <td>${data.ham_count}</td>
                    </tr>
                    <tr>
                      <td><strong>Spam percentage:</strong></td>
                      <td>${data.spam_percentage}</td>
                    </tr>
                  </table>
                `;
                
                // Create chart if we have data
                if (data.total_messages > 0) {
                  const ctx = document.createElement('canvas');
                  document.getElementById('pie-chart').innerHTML = '';
                  document.getElementById('pie-chart').appendChild(ctx);
                  
                  new Chart(ctx, {
                    type: 'pie',
                    data: {
                      labels: ['Spam', 'Ham'],
                      datasets: [{
                        data: [data.spam_count, data.ham_count],
                        backgroundColor: ['#ff4d4d', '#4CAF50'],
                        borderWidth: 1
                      }]
                    },
                    options: {
                      responsive: true,
                      plugins: {
                        legend: {
                          position: 'bottom',
                        }
                      }
                    }
                  });
                }
              });
          }
          
          // Load history with improved confidence visualization
          function loadHistory() {
            fetch('/history')
              .then(response => response.json())
              .then(data => {
                const historySection = document.getElementById('history-content');
                if (data.history.length === 0) {
                  historySection.innerHTML = '<div id="empty-history">No messages analyzed yet</div>';
                } else {
                  let html = '<table>';
                  html += `
                    <tr>
                      <th>Message</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Time</th>
                      <th>Action</th>
                    </tr>
                  `;
                  
                  data.history.forEach((item, index) => {
                    const confidenceValue = parseFloat(item.confidence.replace('%', '')) / 100;
                    
                    // Calculate gradient color based on confidence and prediction
                    let confidenceColor;
                    if (item.prediction === 'Spam') {
                      const redIntensity = Math.min(255, Math.floor(180 + (confidenceValue * 75)));
                      confidenceColor = `rgba(${redIntensity}, 70, 70, 1)`;
                    } else {
                      const greenIntensity = Math.min(255, Math.floor(100 + (confidenceValue * 155)));
                      confidenceColor = `rgba(70, ${greenIntensity}, 70, 1)`;
                    }
                    
                    html += `
                      <tr>
                        <td>${item.message}</td>
                        <td>
                          <div style="display: flex; align-items: center;">
                            <i class="${item.prediction === 'Spam' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle'}" 
                               style="margin-right: 8px; color: ${item.prediction === 'Spam' ? '#ff4d4d' : '#4CAF50'};"></i>
                            ${item.prediction}
                          </div>
                        </td>
                        <td>
                          <div style="width: 100px; background: #f0f0f0; border-radius: 4px; overflow: hidden;">
                            <div style="width: ${confidenceValue * 100}%; height: 8px; background-color: ${confidenceColor};"></div>
                          </div>
                          <div style="font-size: 12px; margin-top: 4px;">${item.confidence}</div>
                        </td>
                        <td class="timestamp">${item.timestamp}</td>
                        <td>
                          <button class="delete-btn" onclick="deleteMessage(${index})">
                            <i class="fas fa-trash"></i>
                          </button>
                        </td>
                      </tr>
                    `;
                  });
                  html += '</table>';
                  historySection.innerHTML = html;
                }
              });
          }
          
          // Delete message
          function deleteMessage(index) {
            fetch('/delete_message', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({index: index})
            })
            .then(response => response.json())
            .then(data => {
              if (data.success) {
                loadHistory(); // Reload the history after deletion
              }
            });
          }
        </script>
      </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Run training script first'}), 500
    
    try:
        message = request.json['message']
        if not isinstance(message, str):
            return jsonify({'error': 'Message must be a string'}), 400
        
        # Process the message
        processed = preprocess_text(message)
        vectorized = vectorizer.transform([processed])
        
        # Get prediction
        y_scores = model.predict_proba(vectorized)[:, 1]
        threshold = 0.7
        score = float(y_scores[0])  # Convert to native Python float
        prediction = 1 if score >= threshold else 0
        result = 'Spam' if prediction == 1 else 'Ham'
        
        # Always show confidence in the predicted class
        # For spam: show spam probability
        # For ham: show ham probability (1 - spam probability)
        confidence_display = f"{score:.2%}" if result == 'Spam' else f"{(1-score):.2%}"
        
        # Raw score for analytics (always the spam probability)
        raw_score = score
        
        # Generate explanation
        explanation = generate_explanation(message, result, confidence_display)
        
        # Update statistics
        stats['total'] += 1
        if result == 'Spam':
            stats['spam'] += 1
        else:
            stats['ham'] += 1
            
        # Store the message and result
        recent_messages.append({
            'message': message,
            'prediction': result,
            'confidence': confidence_display,
            'raw_score': f"{raw_score:.2%}",  # Store raw spam probability for reference
            'explanation': explanation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        logging.info(f"Prediction: {message[:20]}... -> {result} ({confidence_display})")
        
        return jsonify({
            'prediction': result,
            'confidence': confidence_display,
            'raw_score': f"{raw_score:.2%}",  # Return raw spam probability for reference
            'explanation': explanation
        })
    except KeyError:
        return jsonify({'error': 'Missing "message" field'}), 400
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': list(recent_messages)})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'total_messages': stats['total'],
        'spam_count': stats['spam'],
        'ham_count': stats['ham'],
        'spam_percentage': f"{stats['spam']/stats['total']*100:.1f}%" if stats['total'] > 0 else "0%"
    })

@app.route('/delete_message', methods=['POST'])
def delete_message():
    try:
        index = request.json['index']
        if index < 0 or index >= len(recent_messages):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
            
        # Get the message we're about to delete
        message = list(recent_messages)[index]
        
        # Update stats
        if message['prediction'] == 'Spam':
            stats['spam'] -= 1
        else:
            stats['ham'] -= 1
        stats['total'] -= 1
        
        # Convert deque to list, remove item, then create new deque
        message_list = list(recent_messages)
        message_list.pop(index)
        recent_messages.clear()
        for msg in message_list:
            recent_messages.append(msg)
            
        return jsonify({'success': True})
    except KeyError:
        return jsonify({'success': False, 'error': 'Missing index field'}), 400
    except Exception as e:
        logging.error(f"Delete message error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    logging.info("Starting SMS Spam Detection Flask App")
    app.run(debug=True, host='0.0.0.0', port=5001)
