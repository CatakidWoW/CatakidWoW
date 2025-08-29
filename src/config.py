"""
Configuration settings for the Financial AI System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for Financial AI System - Free Data Sources Only"""
    
    # Free Data Sources (No API Keys Required)
    FREE_DATA_SOURCES = {
        'primary': 'yahoo_finance',  # Yahoo Finance - No API key needed
        'backup': 'web_scraping',    # Web scraping from public financial sites
        'news': 'public_news'        # Public financial news sites
    }
    
    # Free Market Data Sources
    MARKET_DATA_SOURCES = {
        'stocks': ['yahoo_finance'],      # Yahoo Finance for stocks
        'forex': ['yahoo_finance'],       # Yahoo Finance for forex
        'crypto': ['yahoo_finance'],      # Yahoo Finance for crypto
        'commodities': ['yahoo_finance'], # Yahoo Finance for commodities
        'indices': ['yahoo_finance']      # Yahoo Finance for indices
    }
    
    # Free News Sources (Public websites)
    FREE_NEWS_SOURCES = {
        'yahoo_finance': 'https://finance.yahoo.com/news/',
        'reuters': 'https://www.reuters.com/markets/',
        'bloomberg': 'https://www.bloomberg.com/markets/',
        'ft': 'https://www.ft.com/markets',
        'cnbc': 'https://www.cnbc.com/markets/',
        'marketwatch': 'https://www.marketwatch.com/markets/',
        'investing': 'https://www.investing.com/news/',
        'seeking_alpha': 'https://seekingalpha.com/news'
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
        'min_data_points': 20,        # Reduced from 100 to 20 for live trading
        'max_price_change': 0.50,     # 50% max price change
        'min_volume': 1000,
        'data_freshness': 300         # 5 minutes max age
    }
    
    # Free Data Rate Limits (Conservative to avoid being blocked)
    FREE_DATA_LIMITS = {
        'yahoo_finance': {
            'requests_per_minute': 30,    # Conservative limit
            'requests_per_hour': 1000,    # Conservative limit
            'delay_between_requests': 2   # 2 seconds between requests
        },
        'web_scraping': {
            'requests_per_minute': 10,    # Very conservative for scraping
            'requests_per_hour': 200,     # Very conservative for scraping
            'delay_between_requests': 6   # 6 seconds between requests
        }
    }
    
    # User Agent for web requests (to avoid being blocked)
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]