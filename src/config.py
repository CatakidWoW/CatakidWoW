"""
Configuration settings for the Financial AI System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Financial AI System"""
    
    # API Keys (set these in your .env file)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    YAHOO_FINANCE_ENABLED = True
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///financial_ai.db')
    
    # Model Configuration
    MODEL_SAVE_PATH = 'models/'
    PREDICTION_HORIZON_DAYS = 30
    TRAINING_LOOKBACK_DAYS = 365
    
    # Risk Management
    MAX_PORTFOLIO_RISK = 0.15  # 15% maximum portfolio risk
    STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE = 0.15  # 15% take profit
    
    # Trading Configuration
    DEFAULT_INITIAL_CAPITAL = 100000  # $100,000
    MAX_POSITION_SIZE = 0.1  # 10% of portfolio per position
    REBALANCE_FREQUENCY_DAYS = 7
    
    # Technical Analysis
    SHORT_TERM_MA = 20
    LONG_TERM_MA = 50
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Sentiment Analysis
    SENTIMENT_ANALYSIS_ENABLED = True
    NEWS_SOURCES = ['reuters', 'bloomberg', 'cnbc', 'yahoo']
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/financial_ai.log'