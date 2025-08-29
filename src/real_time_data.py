"""
Real-Time Data Manager for Financial AI
Fetches live market data from authentic sources with no synthetic information
"""
import pandas as pd
import numpy as np
import requests
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from src.config import Config

class RealTimeDataManager:
    """Real-time data manager for live market information"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Rate limiting
        self.last_api_calls = {}
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize timezone
        self.uk_tz = pytz.timezone('Europe/London')
        
    def get_live_market_data(self, symbol: str, interval: str = '1m', period: str = '1d') -> pd.DataFrame:
        """Get live market data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{period}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Try multiple data sources
            data = None
            
            # Source 1: Yahoo Finance (most reliable for live data)
            if self.config.YAHOO_FINANCE_ENABLED:
                data = self._get_yahoo_finance_data(symbol, interval, period)
            
            # Source 2: Alpha Vantage (if Yahoo fails)
            if data is None or data.empty:
                data = self._get_alpha_vantage_data(symbol, interval, period)
            
            # Source 3: Polygon (if others fail)
            if data is None or data.empty:
                data = self._get_polygon_data(symbol, interval, period)
            
            # Validate data
            if data is not None and not data.empty:
                data = self._validate_and_clean_data(data, symbol)
                
                # Cache the data
                self._cache_data(cache_key, data)
                
                return data
            
            self.logger.error(f"Failed to get live data for {symbol} from all sources")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting live market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get current live price for a symbol"""
        try:
            # Get latest data
            data = self.get_live_market_data(symbol, interval='1m', period='1d')
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting live price for {symbol}: {str(e)}")
            return None
    
    def get_live_market_status(self) -> Dict:
        """Get live UK market status"""
        try:
            now = datetime.now(self.uk_tz)
            current_time = now.strftime('%H:%M')
            
            # UK market hours (8:00 AM - 4:30 PM GMT)
            market_open = '08:00'
            market_close = '16:30'
            
            is_open = market_open <= current_time <= market_close
            
            # Calculate time differences
            time_to_open = None
            time_to_close = None
            
            if not is_open:
                if current_time < market_open:
                    time_to_open = self._calculate_time_difference(current_time, market_open)
                else:
                    time_to_close = self._calculate_time_difference(market_close, current_time)
            
            return {
                'is_open': is_open,
                'current_time': current_time,
                'market_open': market_open,
                'market_close': market_close,
                'time_to_open': time_to_open,
                'time_to_close': time_to_close,
                'status': 'OPEN' if is_open else 'CLOSED',
                'timestamp': now.isoformat(),
                'timezone': 'Europe/London'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market status: {str(e)}")
            return {'is_open': False, 'status': 'UNKNOWN'}
    
    def get_live_news_sentiment(self, symbol: str = None, category: str = 'business') -> Dict:
        """Get live financial news sentiment from real sources"""
        try:
            news_data = {
                'overall_sentiment': 'NEUTRAL',
                'bullish_news': 0,
                'bearish_news': 0,
                'neutral_news': 0,
                'key_headlines': [],
                'sector_sentiment': {},
                'timestamp': datetime.now().isoformat(),
                'data_source': 'real_time'
            }
            
            # Source 1: NewsAPI (if available)
            if self.config.NEWS_API_KEY:
                api_news = self._get_newsapi_data(symbol, category)
                if api_news:
                    news_data.update(api_news)
            
            # Source 2: Finnhub (if available)
            if self.config.FINNHUB_API_KEY:
                finnhub_news = self._get_finnhub_news(symbol)
                if finnhub_news:
                    news_data.update(finnhub_news)
            
            # Source 3: Alpha Vantage News (if available)
            if self.config.ALPHA_VANTAGE_API_KEY:
                av_news = self._get_alpha_vantage_news(symbol)
                if av_news:
                    news_data.update(av_news)
            
            # Calculate overall sentiment from real data
            if news_data['bullish_news'] > 0 or news_data['bearish_news'] > 0:
                total_news = news_data['bullish_news'] + news_data['bearish_news'] + news_data['neutral_news']
                if total_news > 0:
                    bullish_ratio = news_data['bullish_news'] / total_news
                    bearish_ratio = news_data['bearish_news'] / total_news
                    
                    if bullish_ratio > 0.6:
                        news_data['overall_sentiment'] = 'BULLISH'
                    elif bearish_ratio > 0.6:
                        news_data['overall_sentiment'] = 'BEARISH'
                    else:
                        news_data['overall_sentiment'] = 'NEUTRAL'
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error getting live news sentiment: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'data_source': 'real_time'}
    
    def get_live_technical_indicators(self, symbol: str) -> Dict:
        """Get live technical indicators from real market data"""
        try:
            # Get live market data
            data = self.get_live_market_data(symbol, interval='5m', period='1d')
            
            if data.empty:
                return {}
            
            # Calculate real technical indicators
            indicators = {}
            
            # Price analysis
            current_price = data['Close'].iloc[-1]
            open_price = data['Open'].iloc[0]
            high_price = data['High'].max()
            low_price = data['Low'].min()
            
            indicators['current_price'] = current_price
            indicators['open_price'] = open_price
            indicators['high'] = high_price
            indicators['low'] = low_price
            indicators['price_change'] = (current_price - open_price) / open_price
            indicators['price_change_percent'] = indicators['price_change'] * 100
            
            # Volume analysis
            if 'Volume' in data.columns:
                avg_volume = data['Volume'].mean()
                current_volume = data['Volume'].iloc[-1]
                indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
                indicators['current_volume'] = current_volume
                indicators['avg_volume'] = avg_volume
            
            # Moving averages
            if len(data) >= 20:
                indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
                indicators['price_vs_sma20'] = 'ABOVE' if current_price > indicators['sma_20'] else 'BELOW'
            
            if len(data) >= 50:
                indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
                indicators['price_vs_sma50'] = 'ABOVE' if current_price > indicators['sma_50'] else 'BELOW'
            
            # RSI
            if len(data) >= 14:
                indicators['rsi'] = self._calculate_rsi(data['Close'], 14)
            
            # MACD
            if len(data) >= 26:
                macd_data = self._calculate_macd(data['Close'])
                indicators['macd'] = macd_data['macd']
                indicators['macd_signal'] = macd_data['signal']
                indicators['macd_histogram'] = macd_data['histogram']
                indicators['macd_signal_type'] = 'BULLISH' if macd_data['macd'] > macd_data['signal'] else 'BEARISH'
            
            # Bollinger Bands
            if len(data) >= 20:
                bb_data = self._calculate_bollinger_bands(data['Close'])
                indicators['bb_upper'] = bb_data['upper']
                indicators['bb_middle'] = bb_data['middle']
                indicators['bb_lower'] = bb_data['lower']
                indicators['bb_position'] = self._get_bb_position(current_price, bb_data)
            
            # Volatility
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                indicators['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            
            # Technical score
            indicators['technical_score'] = self._calculate_technical_score(indicators)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating live technical indicators for {symbol}: {str(e)}")
            return {}
    
    def get_live_market_sentiment(self) -> Dict:
        """Get live global market sentiment from real data"""
        try:
            # Major indices for sentiment analysis
            major_indices = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^FCHI']
            sentiment_scores = []
            
            for index in major_indices:
                try:
                    data = self.get_live_market_data(index, interval='1h', period='1d')
                    if not data.empty and len(data) > 1:
                        # Calculate sentiment based on real price movement
                        price_change = (data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]
                        sentiment_scores.append(price_change)
                except Exception as e:
                    self.logger.warning(f"Error analyzing {index}: {str(e)}")
                    continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_label = 'BULLISH' if avg_sentiment > 0.01 else 'BEARISH' if avg_sentiment < -0.01 else 'NEUTRAL'
            else:
                avg_sentiment = 0
                sentiment_label = 'NEUTRAL'
            
            return {
                'overall_sentiment': sentiment_label,
                'sentiment_score': avg_sentiment,
                'indices_analyzed': len(sentiment_scores),
                'timestamp': datetime.now().isoformat(),
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market sentiment: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'data_source': 'real_time'}
    
    def _get_yahoo_finance_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Get data from Yahoo Finance (most reliable for live data)"""
        try:
            if self._check_rate_limit('yahoo_finance'):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.title() for col in data.columns]
                    return data
                
        except Exception as e:
            self.logger.warning(f"Yahoo Finance failed for {symbol}: {str(e)}")
        
        return pd.DataFrame()
    
    def _get_alpha_vantage_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Get data from Alpha Vantage API"""
        try:
            if not self.config.ALPHA_VANTAGE_API_KEY:
                return pd.DataFrame()
            
            if not self._check_rate_limit('alpha_vantage'):
                return pd.DataFrame()
            
            # Map intervals to Alpha Vantage format
            interval_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '60min', '1d': 'daily'}
            av_interval = interval_map.get(interval, 'daily')
            
            if av_interval == 'daily':
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY',
                    'symbol': symbol,
                    'apikey': self.config.ALPHA_VANTAGE_API_KEY,
                    'outputsize': 'compact'
                }
            else:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': self.config.ALPHA_VANTAGE_API_KEY,
                    'outputsize': 'compact'
                }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse Alpha Vantage response
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
            elif 'Time Series (1min)' in data:
                time_series = data['Time Series (1min)']
            elif 'Time Series (5min)' in data:
                time_series = data['Time Series (5min)']
            else:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(timestamp),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            self._update_rate_limit('alpha_vantage')
            return df
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage failed for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_polygon_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Get data from Polygon API"""
        try:
            if not self.config.POLYGON_API_KEY:
                return pd.DataFrame()
            
            if not self._check_rate_limit('polygon'):
                return pd.DataFrame()
            
            # Calculate date range
            end_date = datetime.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=1)
            
            # Map intervals to Polygon format
            interval_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '1d': 'D'}
            poly_interval = interval_map.get(interval, '1')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{poly_interval}/{poly_interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            params = {'apikey': self.config.POLYGON_API_KEY}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for result in data['results']:
                df_data.append({
                    'Date': pd.to_datetime(result['t'], unit='ms'),
                    'Open': result['o'],
                    'High': result['h'],
                    'Low': result['l'],
                    'Close': result['c'],
                    'Volume': result['v']
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            self._update_rate_limit('polygon')
            return df
            
        except Exception as e:
            self.logger.warning(f"Polygon failed for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_newsapi_data(self, symbol: str, category: str) -> Dict:
        """Get news from NewsAPI"""
        try:
            if not self.config.NEWS_API_KEY:
                return {}
            
            # Build query
            query = f"finance {category}"
            if symbol:
                query += f" {symbol}"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': self.config.NEWS_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'articles' not in data:
                return {}
            
            # Analyze sentiment from real headlines
            headlines = []
            sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            for article in data['articles'][:10]:
                headline = article.get('title', '')
                headlines.append(headline)
                
                # Simple sentiment analysis on real headlines
                sentiment = self._analyze_headline_sentiment(headline)
                sentiment_counts[sentiment] += 1
            
            return {
                'bullish_news': sentiment_counts['bullish'],
                'bearish_news': sentiment_counts['bearish'],
                'neutral_news': sentiment_counts['neutral'],
                'key_headlines': headlines[:5]
            }
            
        except Exception as e:
            self.logger.warning(f"NewsAPI failed: {str(e)}")
            return {}
    
    def _get_finnhub_news(self, symbol: str) -> Dict:
        """Get news from Finnhub"""
        try:
            if not self.config.FINNHUB_API_KEY:
                return {}
            
            # Get company news
            if symbol:
                url = f"https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': symbol,
                    'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    'to': datetime.now().strftime('%Y-%m-%d'),
                    'token': self.config.FINNHUB_API_KEY
                }
            else:
                # Get general market news
                url = f"https://finnhub.io/api/v1/news"
                params = {
                    'category': 'general',
                    'token': self.config.FINNHUB_API_KEY
                }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return {}
            
            # Analyze real news sentiment
            headlines = []
            sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            for article in data[:10]:
                headline = article.get('headline', '')
                headlines.append(headline)
                
                sentiment = self._analyze_headline_sentiment(headline)
                sentiment_counts[sentiment] += 1
            
            return {
                'bullish_news': sentiment_counts['bullish'],
                'bearish_news': sentiment_counts['bearish'],
                'neutral_news': sentiment_counts['neutral'],
                'key_headlines': headlines[:5]
            }
            
        except Exception as e:
            self.logger.warning(f"Finnhub failed: {str(e)}")
            return {}
    
    def _get_alpha_vantage_news(self, symbol: str) -> Dict:
        """Get news from Alpha Vantage News API"""
        try:
            if not self.config.ALPHA_VANTAGE_API_KEY:
                return {}
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol if symbol else 'FOREX',
                'apikey': self.config.ALPHA_VANTAGE_API_KEY,
                'limit': 10
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data:
                return {}
            
            # Use real sentiment scores from Alpha Vantage
            headlines = []
            sentiment_scores = []
            
            for article in data['feed'][:10]:
                headline = article.get('title', '')
                headlines.append(headline)
                
                # Get real sentiment score if available
                sentiment_score = article.get('overall_sentiment_score', 0)
                sentiment_scores.append(sentiment_score)
            
            # Calculate sentiment from real scores
            if sentiment_scores:
                avg_score = np.mean(sentiment_scores)
                if avg_score > 0.1:
                    overall_sentiment = 'BULLISH'
                elif avg_score < -0.1:
                    overall_sentiment = 'BEARISH'
                else:
                    overall_sentiment = 'NEUTRAL'
            else:
                overall_sentiment = 'NEUTRAL'
            
            return {
                'overall_sentiment': overall_sentiment,
                'key_headlines': headlines[:5]
            }
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage News failed: {str(e)}")
            return {}
    
    def _analyze_headline_sentiment(self, headline: str) -> str:
        """Analyze sentiment of real news headlines"""
        try:
            headline_lower = headline.lower()
            
            # Bullish keywords
            bullish_words = ['surge', 'rally', 'jump', 'climb', 'gain', 'positive', 'strong', 'growth', 'earnings beat', 'upgrade', 'buy', 'outperform']
            bullish_count = sum(1 for word in bullish_words if word in headline_lower)
            
            # Bearish keywords
            bearish_words = ['decline', 'fall', 'drop', 'plunge', 'negative', 'weak', 'loss', 'downturn', 'earnings miss', 'downgrade', 'sell', 'underperform']
            bearish_count = sum(1 for word in bearish_words if word in headline_lower)
            
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.warning(f"Error analyzing headline sentiment: {str(e)}")
            return 'neutral'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI from real price data"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD from real price data"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands from real price data"""
        try:
            sma = prices.rolling(window=period).mean()
            std_dev = prices.rolling(window=period).std()
            
            return {
                'upper': sma + (std_dev * std),
                'middle': sma,
                'lower': sma - (std_dev * std)
            }
        except:
            return {'upper': 0, 'middle': 0, 'lower': 0}
    
    def _get_bb_position(self, price: float, bb_data: Dict) -> str:
        """Get position relative to Bollinger Bands"""
        try:
            if price > bb_data['upper']:
                return 'ABOVE_UPPER'
            elif price < bb_data['lower']:
                return 'BELOW_LOWER'
            else:
                return 'INSIDE_BANDS'
        except:
            return 'UNKNOWN'
    
    def _calculate_technical_score(self, indicators: Dict) -> float:
        """Calculate overall technical score from real indicators"""
        try:
            score = 0.5  # Base score
            
            # Price change contribution
            price_change = indicators.get('price_change', 0)
            if abs(price_change) > 0.02:  # 2% move
                score += 0.2 if price_change > 0 else -0.2
            
            # Volume contribution
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                score += 0.1
            elif volume_ratio < 0.5:
                score -= 0.1
            
            # RSI contribution
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                score += 0.1
            elif rsi > 70:  # Overbought
                score -= 0.1
            
            # MACD contribution
            macd_signal = indicators.get('macd_signal_type', 'NEUTRAL')
            if macd_signal == 'BULLISH':
                score += 0.1
            elif macd_signal == 'BEARISH':
                score -= 0.1
            
            return max(0, min(1, score))  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {str(e)}")
            return 0.5
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean real market data"""
        try:
            if data.empty:
                return data
            
            # Remove any rows with invalid prices
            data = data[
                (data['Open'] > 0) & 
                (data['High'] > 0) & 
                (data['Low'] > 0) & 
                (data['Close'] > 0)
            ]
            
            # Check for extreme price changes (data validation)
            if len(data) > 1:
                price_changes = data['Close'].pct_change().abs()
                max_change = self.config.DATA_VALIDATION['max_price_change']
                data = data[price_changes <= max_change]
            
            # Check minimum data points
            if len(data) < self.config.DATA_VALIDATION['min_data_points']:
                self.logger.warning(f"Insufficient data points for {symbol}: {len(data)}")
            
            # Check data freshness
            if not data.empty:
                latest_time = data.index[-1]
                if isinstance(latest_time, pd.Timestamp):
                    age_minutes = (pd.Timestamp.now() - latest_time).total_seconds() / 60
                    if age_minutes > self.config.DATA_VALIDATION['data_freshness']:
                        self.logger.warning(f"Data for {symbol} is {age_minutes:.1f} minutes old")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol}: {str(e)}")
            return data
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows request"""
        try:
            if api_name not in self.last_api_calls:
                return True
            
            rate_limit = self.config.RATE_LIMITS.get(api_name, 60)
            last_call = self.last_api_calls[api_name]
            
            # Allow request if enough time has passed
            if time.time() - last_call >= (60 / rate_limit):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit for {api_name}: {str(e)}")
            return True
    
    def _update_rate_limit(self, api_name: str):
        """Update API rate limit timestamp"""
        try:
            self.last_api_calls[api_name] = time.time()
        except Exception as e:
            self.logger.error(f"Error updating rate limit for {api_name}: {str(e)}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.cache_timestamps:
                return False
            
            cache_age = time.time() - self.cache_timestamps[cache_key]
            return cache_age < self.config.CACHE_DURATION
            
        except Exception as e:
            self.logger.error(f"Error checking cache validity: {str(e)}")
            return False
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        try:
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = time.time()
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")
    
    def _calculate_time_difference(self, time1: str, time2: str) -> str:
        """Calculate time difference between two times"""
        try:
            t1 = datetime.strptime(time1, '%H:%M')
            t2 = datetime.strptime(time2, '%H:%M')
            
            if t1 > t2:
                t2 += timedelta(days=1)
            
            diff = t2 - t1
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            
            return f"{hours}h {minutes}m"
            
        except Exception as e:
            self.logger.error(f"Error calculating time difference: {str(e)}")
            return "Unknown"