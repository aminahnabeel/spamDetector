import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re
import time

print("🚀 Starting Fast Spam Detection Model Training...")

# Load the pre-processed datasets
print("📁 Loading datasets...")
train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")

# Keep only required columns
train_df = train_df[["message", "spam"]]
test_df = test_df[["message", "spam"]]

print("📊 Dataset distribution:")
print("Train spam distribution:", train_df['spam'].value_counts().to_dict())
print("Test spam distribution:", test_df['spam'].value_counts().to_dict())

def preprocess_text(text):
    """Enhanced text preprocessing for spam detection"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Keep URLs as features (important for spam detection)
    text = re.sub(r'http[s]?://\S+', 'URL_TOKEN', text)
    text = re.sub(r'www\.\S+', 'URL_TOKEN', text)
    
    # Keep phone numbers as features (important for spam detection)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}', 'PHONE_TOKEN', text)
    text = re.sub(r'\b\d{10,11}\b', 'PHONE_TOKEN', text)
    
    # Keep email patterns
    text = re.sub(r'\S+@\S+', 'EMAIL_TOKEN', text)
    
    # Keep money/price patterns
    text = re.sub(r'\$\d+', 'MONEY_TOKEN', text)
    text = re.sub(r'\d+\s*rupee', 'MONEY_TOKEN', text)
    
    return text.strip()

# Preprocess text data
print("🔄 Preprocessing text data...")
train_df['processed_message'] = train_df['message'].apply(preprocess_text)
test_df['processed_message'] = test_df['message'].apply(preprocess_text)

# Create TF-IDF features
print("📝 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,  # Top 10k features
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,            # Ignore terms that appear in less than 2 documents
    max_df=0.95,         # Ignore terms that appear in more than 95% of documents
    strip_accents='unicode',
    lowercase=True,
    stop_words=None      # Keep all words for spam detection
)

X_train = vectorizer.fit_transform(train_df['processed_message'])
X_test = vectorizer.transform(test_df['processed_message'])
y_train = train_df['spam']
y_test = test_df['spam']

print(f"✅ Features created: {X_train.shape[1]} features")

# Train multiple fast models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42, probability=True)
}

best_model = None
best_score = 0
best_name = ""

print("\n🏋️ Training multiple models...")
for name, model in models.items():
    print(f"\n📈 Training {name}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    print(f"✅ {name} - Accuracy: {accuracy:.4f} - Time: {training_time:.2f}s")
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Keep track of best model
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} with accuracy: {best_score:.4f}")

# Save the best model and vectorizer
print("💾 Saving the best model...")
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Create a simple prediction function
def predict_spam(message):
    """Predict if a message is spam"""
    # Load models if not already loaded
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    
    # Preprocess and vectorize
    processed = preprocess_text(message)
    features = vec.transform([processed])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'is_spam': bool(prediction),
        'spam_probability': float(probability[1]),
        'confidence': float(max(probability))
    }

# Test with some examples
print("\n🧪 Testing with sample messages...")
test_messages = [
    "Hi, how are you today?",
    "Click here for free money: http://freecash.com/signup",
    "Call 03001234567 for amazing offers!",
    "Meeting tomorrow at 3 PM",
    "You are such a pathetic loser",
    "Congratulations! You've won a lottery!"
]

for msg in test_messages:
    result = predict_spam(msg)
    spam_status = "🚨 SPAM" if result['is_spam'] else "✅ HAM"
    print(f"{spam_status} ({result['confidence']:.2%}): {msg[:50]}...")

print("\n✅ Fast spam detection model trained successfully!")
print(f"📁 Model saved as: spam_model.pkl")
print(f"📁 Vectorizer saved as: vectorizer.pkl")
print(f"⚡ Total training time: Much faster than BERT!")
