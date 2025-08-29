"""
Financial Data Manager
Handles data fetching, processing, and storage for the Financial AI System
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from src.config import Config

class FinancialDataManager:
    """Manages financial data from multiple sources"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            cache_key = f"{symbol}_{period}_{interval}"
            self.cache[cache_key] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks simultaneously
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
        
        Returns:
            Dictionary mapping symbols to their data
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if not data.empty:
                data_dict[symbol] = data
                
        return data_dict
    
    def get_market_data(self, indices: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch major market indices data
        
        Args:
            indices: List of market indices (default: major indices)
        
        Returns:
            Dictionary mapping indices to their data
        """
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        
        return self.get_multiple_stocks(indices)
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information and fundamentals
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0)
            }
            
            return company_info
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            data: OHLCV DataFrame
        
        Returns:
            DataFrame with technical indicators added
        """
        if data.empty:
            return data
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=self.config.SHORT_TERM_MA).mean()
        data['SMA_50'] = data['Close'].rolling(window=self.config.LONG_TERM_MA).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'], self.config.RSI_PERIOD)
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price changes
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def get_earnings_calendar(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get upcoming earnings calendar
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with earnings information
        """
        try:
            # This would typically use a paid API like Alpha Vantage
            # For now, we'll return a placeholder
            self.logger.info("Earnings calendar feature requires premium API access")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {str(e)}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Get news sentiment for a stock
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
        
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # This would integrate with news APIs and sentiment analysis
            # For now, return placeholder data
            sentiment_data = {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'last_updated': datetime.now().isoformat()
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment for {symbol}: {str(e)}")
            return {}
    
    def export_data(self, data: pd.DataFrame, filename: str, format: str = 'csv') -> bool:
        """
        Export data to various formats
        
        Args:
            data: DataFrame to export
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'csv':
                data.to_csv(filename)
            elif format.lower() == 'excel':
                data.to_excel(filename, engine='openpyxl')
            elif format.lower() == 'json':
                data.to_json(filename)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Data exported successfully to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return False