'''Model Loading: pipeline('sentiment-analysis') initializes a pre-trained sentiment analysis model to classify text into sentiment categories with confidence scores.

Sentiment Analysis Function: analyze_sentiment(text) uses the model to get sentiment labels and scores for the given text input.

Fraud Detection Logic: detect_fraudulent_behavior(messages) scans each message for predefined suspicious phrases and flags them if detected.

Sentiment Reporting: The script prints each message’s sentiment label and score, indicating its emotional tone and confidence level.
'''


from transformers import pipeline

# Load the sentiment analysis model from Hugging Face
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text and return the sentiment label and score.
    """
    result = sentiment_analyzer(text)
    return result

def detect_fraudulent_behavior(messages):
    """
    Analyze a list of chat messages to detect potential fraudulent behavior based on sentiment analysis.
    """
    suspicious_phrases = [
        "I am the customer",
        "I have the device",
        "My account",
        "My device",
        "I am not the customer",
        "I'm just trying to test",
        "I know the device"
    ]
    
    for message in messages:
        sentiment_result = analyze_sentiment(message)
        sentiment_label = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
        
        # Check if the message contains suspicious phrases
        if any(phrase.lower() in message.lower() for phrase in suspicious_phrases):
            print(f"Suspicious message detected: {message}")
            print(f"Sentiment: {sentiment_label} (Score: {sentiment_score})")
        else:
            print(f"Message: {message}")
            print(f"Sentiment: {sentiment_label} (Score: {sentiment_score})")

# Example chat messages
chat_messages = [
    "I am the customer and I need help with my account.",
    "I have the device, but it seems to be malfunctioning.",
    "Can you please assist me? I am just trying to test something.",
    "This is not related to my account at all.",
]

# Analyze chat messages
detect_fraudulent_behavior(chat_messages)
