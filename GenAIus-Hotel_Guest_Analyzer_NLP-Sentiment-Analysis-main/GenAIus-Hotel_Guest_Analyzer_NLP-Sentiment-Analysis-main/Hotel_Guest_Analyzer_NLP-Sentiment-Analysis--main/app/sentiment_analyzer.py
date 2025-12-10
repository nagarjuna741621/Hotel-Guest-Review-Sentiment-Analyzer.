"""
Sentiment Analysis Module
"""
import re
from textblob import TextBlob

def prepare_text(text):
    """
    Task 1: Data Collection and Preparation
    Clean and prepare text for sentiment analysis
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Cleaned and prepared text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation, keep only alphanumeric and whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment(text):
    """
    Task 2: Sentiment Analysis
    Analyze sentiment of a review using TextBlob
    
    Args:
        text (str): Review text (can be raw or prepared)
        
    Returns:
        dict: Dictionary containing sentiment label and polarity score
    """
    # Prepare the text
    prepared_text = prepare_text(text)
    
    # Create TextBlob object
    blob = TextBlob(prepared_text)
    
    # Get polarity (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Classify sentiment
    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'polarity': round(polarity, 3),
        'subjectivity': round(blob.sentiment.subjectivity, 3)
    }

def analyze_batch(reviews):
    """
    Analyze multiple reviews at once
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        list: List of analysis results
    """
    results = []
    for review in reviews:
        if review.strip():  # Skip empty reviews
            result = analyze_sentiment(review)
            result['review'] = review
            results.append(result)
    return results

def get_sentiment_distribution(analyses):
    """
    Task 3: Calculate sentiment distribution
    
    Args:
        analyses (list): List of analysis results
        
    Returns:
        dict: Distribution of sentiments
    """
    distribution = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    
    for analysis in analyses:
        sentiment = analysis.get('sentiment', 'neutral')
        distribution[sentiment] = distribution.get(sentiment, 0) + 1
    
    return distribution




