"""
Environment Setup Script for Hotel Guest Sentiment Analyzer
Run this script to verify and set up the project environment
"""
import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("[ERROR] Python 3.7+ is required")
        return False
    print("[OK] Python version is compatible")
    return True

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = {
        'pandas': 'pandas>=1.3.0',
        'numpy': 'numpy>=1.21.0',
        'matplotlib': 'matplotlib>=3.4.0',
        'seaborn': 'seaborn>=0.11.0',
        'textblob': 'textblob>=0.17.0',
        'flask': 'Flask>=2.0.0',
        'flask_login': 'Flask-Login>=0.6.0',
        'flask_sqlalchemy': 'Flask-SQLAlchemy>=2.5.0',
        'werkzeug': 'Werkzeug>=2.0.0',
        'nltk': 'nltk>=3.6.0'
    }
    
    print("\n" + "="*60)
    print("Checking Required Packages")
    print("="*60)
    
    missing_packages = []
    
    for package_name, package_spec in required_packages.items():
        try:
            __import__(package_name)
            print(f"[OK] {package_name} is installed")
        except ImportError:
            print(f"[MISSING] {package_name} is missing")
            missing_packages.append(package_spec)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"   Installing {package}...")
            if install_package(package):
                print(f"   [OK] {package} installed successfully")
            else:
                print(f"   [ERROR] Failed to install {package}")
        return len(missing_packages) == 0
    else:
        print("\n[OK] All required packages are installed!")
        return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\n" + "="*60)
    print("Downloading NLTK Data")
    print("="*60)
    
    try:
        import nltk
        nltk_data = ['punkt', 'stopwords', 'vader_lexicon']
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                print(f"[OK] {data} is already downloaded")
            except LookupError:
                print(f"[DOWNLOAD] Downloading {data}...")
                nltk.download(data, quiet=True)
                print(f"[OK] {data} downloaded successfully")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error downloading NLTK data: {e}")
        return False

def check_dataset():
    """Check if dataset exists"""
    print("\n" + "="*60)
    print("Checking Dataset")
    print("="*60)
    
    dataset_path = 'data/hotel_reviews_dataset.csv'
    if os.path.exists(dataset_path):
        import pandas as pd
        try:
            df = pd.read_csv(dataset_path)
            print(f"[OK] Dataset found: {dataset_path}")
            print(f"   Total reviews: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"[ERROR] Error reading dataset: {e}")
            return False
    else:
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n" + "="*60)
    print("Creating Directories")
    print("="*60)
    
    directories = ['data', 'instance', 'static/css', 'templates', 'app']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[OK] Created directory: {directory}")
        else:
            print(f"[OK] Directory exists: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("="*60)
    print("Hotel Guest Sentiment Analyzer - Environment Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check and install packages
    if not check_and_install_packages():
        print("\n[WARNING] Some packages failed to install. Please install manually:")
        print("   pip install -r requirements.txt")
    
    # Download NLTK data
    download_nltk_data()
    
    # Check dataset
    check_dataset()
    
    # Create directories
    create_directories()
    
    print("\n" + "="*60)
    print("Environment Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run Flask app: python run.py")
    print("2. Open notebook: hotel_sentiment_analysis.ipynb")
    print("3. Access web app: http://localhost:5000")
    print("\n[SUCCESS] You're all set!")

if __name__ == '__main__':
    main()

