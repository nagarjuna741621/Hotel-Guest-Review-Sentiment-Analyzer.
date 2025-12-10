# Hotel Guest Sentiment Analyzer - Project Structure

## 📁 Project Organization

```
Hotel_Guest_Analyser_NLP/
│
├── app/                          # Flask application package
│   ├── __init__.py              # App factory and initialization
│   ├── routes.py                # Application routes (including dataset analysis)
│   ├── models.py                # Database models (User, Analysis)
│   ├── auth.py                  # Authentication helpers
│   └── sentiment_analyzer.py    # Core sentiment analysis logic
│
├── templates/                    # Jinja2 HTML templates
│   ├── base.html               # Base template
│   ├── index.html              # Home page
│   ├── login.html              # Login page
│   ├── signup.html             # Registration page
│   ├── dashboard.html          # User dashboard
│   ├── analyze.html            # Single/batch review analysis
│   └── dataset_analysis.html   # Real dataset analysis results
│
├── static/                      # Static files
│   └── css/
│       └── style.css           # Custom styles
│
├── data/                        # Dataset files
│   └── hotel_reviews_dataset.csv  # Real-world hotel reviews (56 reviews)
│
├── instance/                     # Instance-specific files
│   └── database.db             # SQLite database (auto-generated)
│
├── hotel_sentiment_analysis.ipynb  # Jupyter notebook for analysis
│
├── app.py                       # Application entry point (alternative)
├── run.py                       # Main application runner
├── config.py                   # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── PROJECT_STRUCTURE.md        # This file
```

## 🎯 Key Features

### 1. **Jupyter Notebook** (`hotel_sentiment_analysis.ipynb`)
   - Complete data analysis workflow
   - Data cleaning and preprocessing
   - Sentiment analysis using TextBlob
   - Comprehensive visualizations
   - Aspect analysis
   - Model accuracy evaluation

### 2. **Flask Web Application**
   - **Authentication System**: User signup, login, logout
   - **Single Review Analysis**: Analyze individual hotel reviews
   - **Batch Analysis**: Analyze multiple reviews at once
   - **Real Dataset Analysis**: Analyze the complete 56-review dataset
   - **Dashboard**: User statistics and recent analyses
   - **Database**: Stores analysis history per user

### 3. **Real-World Dataset**
   - 56 hotel reviews with actual sentiment labels
   - Primary aspects identified for each review
   - Ready for sentiment prediction and evaluation

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Flask Application**:
   ```bash
   python run.py
   ```
   Access at: http://localhost:5000

3. **Run Jupyter Notebook**:
   - Open `hotel_sentiment_analysis.ipynb` in Jupyter
   - Run all cells sequentially

## 📊 Dataset Information

- **File**: `data/hotel_reviews_dataset.csv`
- **Total Reviews**: 56
- **Columns**: Review ID, Sentiment, Primary Aspect, Cleaned Text
- **Sentiment Distribution**:
  - Positive: 24 (42.86%)
  - Negative: 24 (42.86%)
  - Mixed: 8 (14.29%)

## 🔧 Technology Stack

- **Backend**: Flask, Flask-Login, Flask-SQLAlchemy
- **NLP**: TextBlob, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Chart.js
- **Frontend**: Bootstrap 5, Jinja2
- **Database**: SQLite

## 📝 Notes

- The application uses TextBlob for sentiment analysis
- Model accuracy on the dataset: ~76.79%
- All analysis results are saved to the database
- The notebook generates comprehensive visualizations and CSV exports


