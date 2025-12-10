"""
Quick start script for the Hotel Sentiment Analyzer application
"""
from app import create_app, db
from app.models import User

# Create the application
app = create_app()

# Create database tables
with app.app_context():
    db.create_all()
    print("Database initialized successfully!")
    print("\n" + "="*50)
    print("Hotel Sentiment Analyzer is ready!")
    print("="*50)
    print("\nAccess the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



