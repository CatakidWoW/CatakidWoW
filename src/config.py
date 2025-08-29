"""
Configuration settings for the Financial AI System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for Financial AI System"""
    
    # Real-time API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    YAHOO_FINANCE_ENABLED = os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', '')
    
    # Real-time News Sources
    NEWS_SOURCES = {
        'newsapi': 'https://newsapi.org/v2/',
        'finnhub': 'https://finnhub.io/api/v1/',
        'polygon': 'https://api.polygon.io/v2/',
        'alpha_vantage': 'https://www.alphavantage.co/query',
        'yahoo_finance': 'https://query2.finance.yahoo.com/v8/finance/',
        'reuters': 'https://www.reuters.com/markets/',
        'bloomberg': 'https://www.bloomberg.com/markets/',
        'ft': 'https://www.ft.com/markets/',
        'cnbc': 'https://www.cnbc.com/markets/',
        'marketwatch': 'https://www.marketwatch.com/markets/'
    }
    
    # Real-time Market Data
    MARKET_DATA_SOURCES = {
        'stocks': ['yahoo_finance', 'alpha_vantage', 'polygon', 'iex_cloud'],
        'forex': ['alpha_vantage', 'polygon', 'twelve_data'],
        'crypto': ['alpha_vantage', 'polygon', 'yahoo_finance'],
        'commodities': ['alpha_vantage', 'polygon', 'yahoo_finance'],
        'indices': ['yahoo_finance', 'alpha_vantage', 'polygon']
    }
    
    # Real-time Data Intervals
    REAL_TIME_INTERVALS = {
        'live': '1m',      # 1 minute for live trading
        'intraday': '5m',  # 5 minutes for intraday analysis
        'daily': '1d',     # Daily for trend analysis
        'weekly': '1wk'    # Weekly for long-term analysis
    }
    
    # Database and Storage
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///financial_ai.db')
    MODEL_SAVE_PATH = 'models/'
    CACHE_DURATION = 300  # 5 minutes cache for real-time data
    
    # Risk Management (Real-time)
    MAX_PORTFOLIO_RISK = 0.15
    DEFAULT_INITIAL_CAPITAL = 147.0  # £147 capital
    MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
    MIN_RISK_REWARD_RATIO = 2.0
    
    # Trading Parameters (Real-time)
    SHORT_TERM_MA = 20
    LONG_TERM_MA = 50
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Real-time Market Hours (UK)
    UK_MARKET_HOURS = {
        'open': '08:00',
        'close': '16:30',
        'timezone': 'Europe/London'
    }
    
    # Real-time Update Frequencies
    UPDATE_FREQUENCIES = {
        'market_status': 60,      # 1 minute
        'price_data': 30,         # 30 seconds
        'news_sentiment': 300,    # 5 minutes
        'technical_indicators': 60, # 1 minute
        'risk_metrics': 120       # 2 minutes
    }
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/financial_ai.log'
    
    # Real-time Data Validation
    DATA_VALIDATION = {
        'min_data_points': 100,
        'max_price_change': 0.50,  # 50% max price change
        'min_volume': 1000,
        'data_freshness': 300      # 5 minutes max age
    }
    
    # API Rate Limits
    RATE_LIMITS = {
        'alpha_vantage': 5,       # 5 calls per minute
        'polygon': 5,             # 5 calls per minute
        'finnhub': 60,            # 60 calls per minute
        'newsapi': 100,           # 100 calls per day
        'yahoo_finance': 2000     # 2000 calls per hour
    }