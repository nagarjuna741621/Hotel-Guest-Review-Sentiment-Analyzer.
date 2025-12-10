"""
Application routes - MODIFIED FOR VADER & ML HYBRID
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User, Analysis
from app.auth import create_user, get_user_by_username
# Import VADER analysis and ML prediction functions
from app.sentiment_analyzer import analyze_review_vader, predict_sentiment_ml
from werkzeug.security import check_password_hash
import json
import pandas as pd
import os
from collections import Counter

bp = Blueprint('routes', __name__)

# Helper to get VADER distribution (VADER uses 'neutral', not 'mixed')
def get_vader_distribution(analyses):
    """
    Calculate VADER sentiment distribution
    """
    distribution = {
        'positive': 0,
        'negative': 0,
        'neutral': 0 # For VADER's mixed/neutral class
    }
    
    for analysis in analyses:
        # Use 'sentiment' key from VADER result
        sentiment = analysis.get('sentiment', 'neutral')
        distribution[sentiment] = distribution.get(sentiment, 0) + 1
    
    return distribution

# --- Standard Auth/Index Routes (No Change Needed) ---

@bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    # ... (Authentication code remains the same)
    if current_user.is_authenticated:
        return redirect(url_for('routes.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('signup.html')
        
        # Create user
        user, error = create_user(username, email, password)
        if error:
            flash(error, 'error')
            return render_template('signup.html')
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('routes.login'))
    
    return render_template('signup.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    # ... (Authentication code remains the same)
    if current_user.is_authenticated:
        return redirect(url_for('routes.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        user = get_user_by_username(username)
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('routes.dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('routes.index'))

@bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard - MODIFIED to use compound_score"""
    # Get user's recent analyses
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id)\
        .order_by(Analysis.created_at.desc())\
        .limit(10)\
        .all()
    
    # Get statistics
    total_analyses = Analysis.query.filter_by(user_id=current_user.id).count()
    positive_count = Analysis.query.filter_by(user_id=current_user.id, sentiment='positive').count()
    negative_count = Analysis.query.filter_by(user_id=current_user.id, sentiment='negative').count()
    neutral_count = Analysis.query.filter_by(user_id=current_user.id, sentiment='neutral').count()
    
    stats = {
        'total': total_analyses,
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count
    }
    
    return render_template('dashboard.html', 
                         recent_analyses=recent_analyses,
                         stats=stats)

@bp.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Sentiment analysis page - MODIFIED to use VADER"""
    if request.method == 'POST':
        review_text = request.form.get('review_text', '').strip()
        batch_reviews = request.form.get('batch_reviews', '').strip()
        
        if batch_reviews:
            # Batch analysis using VADER
            reviews = [r.strip() for r in batch_reviews.split('\n') if r.strip()]
            results = [analyze_review_vader(r) for r in reviews]
            
            # Re-associate review text
            for i, result in enumerate(results):
                result['review'] = reviews[i]

            # Save to database
            for result in results:
                analysis = Analysis(
                    user_id=current_user.id,
                    review_text=result['review'],
                    sentiment=result['sentiment'],
                    compound_score=result['compound_score'] # Use new column
                )
                db.session.add(analysis)
            db.session.commit()
            
            # Calculate distribution
            distribution = get_vader_distribution(results)
            
            return render_template('analyze.html', 
                                 results=results,
                                 distribution=distribution,
                                 is_batch=True)
        
        elif review_text:
            # Single review analysis using VADER
            result = analyze_review_vader(review_text)
            result['review'] = review_text
            
            # Save to database
            analysis = Analysis(
                user_id=current_user.id,
                review_text=review_text,
                sentiment=result['sentiment'],
                compound_score=result['compound_score'] # Use new column
            )
            db.session.add(analysis)
            db.session.commit()
            
            # Create complete distribution dictionary
            distribution = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            distribution[result['sentiment']] = 1
            
            return render_template('analyze.html', 
                                 results=[result],
                                 distribution=distribution,
                                 is_batch=False)
        else:
            flash('Please enter a review to analyze', 'error')
    
    return render_template('analyze.html')

@bp.route('/analyze-dataset', methods=['GET', 'POST'])
@login_required
def analyze_dataset():
    """Analyze the real-world hotel reviews dataset - MODIFIED for VADER and ML"""
    dataset_path = os.path.join(current_app.root_path, '..', 'data', 'hotel_reviews_dataset.csv')
    
    if not os.path.exists(dataset_path):
        flash('Dataset file not found. Ensure it is in the data/ folder.', 'error')
        return redirect(url_for('routes.dashboard'))
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Prepare reviews for analysis (use the correct column)
        reviews = df['Cleaned Text (Lowercased)'].dropna().astype(str).tolist()
        
        # 1. VADER Analysis (List of dicts)
        vader_results = [analyze_review_vader(r) for r in reviews]
        
        # 2. ML Prediction (List of labels)
        ml_predictions = predict_sentiment_ml(reviews)

        # Combine results and prepare for template
        combined_results = []
        actual_labels = df['Sentiment'].apply(lambda x: x.lower().replace('mixed', 'neutral')) # Map for accuracy
        
        correct_vader = 0
        correct_ml = 0
        total = 0
        
        for i, v_result in enumerate(vader_results):
            # Combine VADER result with ML prediction and metadata
            c_result = v_result
            c_result['review'] = reviews[i]
            
            if i < len(df):
                c_result['review_id'] = int(df.iloc[i]['Review ID'])
                c_result['actual_sentiment'] = df.iloc[i]['Sentiment']
                c_result['primary_aspect'] = df.iloc[i]['Primary Aspect']
            
            # Add ML prediction
            c_result['ml_prediction'] = ml_predictions[i]
            
            # Calculate accuracy on mapped actual labels
            actual_mapped = actual_labels.iloc[i] 
            
            if c_result['sentiment'] == actual_mapped: # VADER accuracy
                correct_vader += 1
            
            if c_result['ml_prediction'] == actual_mapped: # ML accuracy
                correct_ml += 1
                
            total += 1
            combined_results.append(c_result)

        # Calculate statistics
        vader_distribution = get_vader_distribution(combined_results)
        
        accuracy_vader = (correct_vader / total * 100) if total > 0 else 0
        accuracy_ml = (correct_ml / total * 100) if total > 0 else 0

        # Aspect analysis (Retain original logic)
        all_aspects = []
        for aspect_str in df['Primary Aspect'].dropna():
            aspects = [a.strip() for a in str(aspect_str).split('&')]
            all_aspects.extend(aspects)
        aspect_counts = Counter(all_aspects)
        top_aspects = dict(aspect_counts.most_common(10))
        
        # Compound score statistics
        compound_scores = [r['compound_score'] for r in combined_results]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        
        # Save sample to database (using VADER scores)
        for result in combined_results[:10]:  # Save first 10
            analysis = Analysis(
                user_id=current_user.id,
                review_text=result['review'][:500],
                sentiment=result['sentiment'],
                compound_score=result['compound_score']
            )
            db.session.add(analysis)
        db.session.commit()
        
        return render_template('dataset_analysis.html',
                             results=combined_results,
                             distribution=vader_distribution,
                             accuracy_vader=round(accuracy_vader, 2), # Pass VADER accuracy
                             accuracy_ml=round(accuracy_ml, 2),     # Pass ML accuracy
                             total_reviews=len(reviews),
                             top_aspects=top_aspects,
                             avg_compound=round(avg_compound, 3)) # Pass VADER avg
    
    except Exception as e:
        flash(f'Error analyzing dataset: {str(e)}', 'error')
        # This is a critical point, may mean the ML model file is missing
        flash('CRITICAL: Check if hotel_sentiment_analysis.ipynb has been run to train/save the ML model.', 'error')
        return redirect(url_for('routes.dashboard'))

@bp.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    """API endpoint for sentiment analysis - MODIFIED to use VADER"""
    data = request.get_json()
    review_text = data.get('text', '').strip()
    
    if not review_text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyze_review_vader(review_text) # Use VADER
    
    # Save to database
    analysis = Analysis(
        user_id=current_user.id,
        review_text=review_text,
        sentiment=result['sentiment'],
        compound_score=result['compound_score'] # Use new column
    )
    db.session.add(analysis)
    db.session.commit()
    
    return jsonify(result)