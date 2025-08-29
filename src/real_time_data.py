"""
Real-Time Data Manager for Financial AI
Fetches live market data from FREE sources with no API keys required
"""
import pandas as pd
import numpy as np
import requests
import logging
import time
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from bs4 import BeautifulSoup
from src.config import Config

class RealTimeDataManager:
    """Real-time data manager for live market information using FREE sources only"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Rotate user agents to avoid being blocked
        self.current_user_agent = random.choice(self.config.USER_AGENTS)
        self.session.headers.update({
            'User-Agent': self.current_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Rate limiting for free sources
        self.last_request_time = {}
        self.request_count = {}
        
        # Cache system
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize timezone
        self.uk_tz = pytz.timezone('Europe/London')
        
    def get_live_market_data(self, symbol: str, interval: str = '1m', period: str = '1d') -> pd.DataFrame:
        """Get live market data from FREE sources (no API key needed)"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}_{period}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Try Yahoo Finance first (primary free source)
            data = self._get_yahoo_finance_data(symbol, interval, period)
            
            # If Yahoo Finance fails, try web scraping as backup
            if data is None or data.empty:
                data = self._get_web_scraped_data(symbol)
            
            # Validate data
            if data is not None and not data.empty:
                data = self._validate_and_clean_data(data, symbol)
                
                # Cache the data
                self._cache_data(cache_key, data)
                
                return data
            
            self.logger.error(f"Failed to get live data for {symbol} from free sources")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting live market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get current live price for a symbol from free sources"""
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
        """Get live UK market status using free time checking"""
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
                'timezone': 'Europe/London',
                'data_source': 'free_time_check'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market status: {str(e)}")
            return {'is_open': False, 'status': 'UNKNOWN', 'data_source': 'free_time_check'}
    
    def get_live_news_sentiment(self, symbol: str = None, category: str = 'business') -> Dict:
        """Get live financial news sentiment from FREE public sources"""
        try:
            news_data = {
                'overall_sentiment': 'NEUTRAL',
                'bullish_news': 0,
                'bearish_news': 0,
                'neutral_news': 0,
                'key_headlines': [],
                'sector_sentiment': {},
                'timestamp': datetime.now().isoformat(),
                'data_source': 'free_public_sources'
            }
            
            # Get news from free public sources
            if symbol:
                # Get symbol-specific news
                symbol_news = self._get_symbol_news_from_free_sources(symbol)
                if symbol_news:
                    news_data.update(symbol_news)
            else:
                # Get general market news
                general_news = self._get_general_news_from_free_sources()
                if general_news:
                    news_data.update(general_news)
            
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
            return {'overall_sentiment': 'NEUTRAL', 'data_source': 'free_public_sources'}
    
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
                'data_source': 'free_yahoo_finance'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting live market sentiment: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'data_source': 'free_yahoo_finance'}
    
    def _get_yahoo_finance_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Get data from Yahoo Finance (FREE - no API key needed)"""
        try:
            if self._check_rate_limit('yahoo_finance'):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.title() for col in data.columns]
                    
                    # Update rate limit
                    self._update_rate_limit('yahoo_finance')
                    
                    return data
                
        except Exception as e:
            self.logger.warning(f"Yahoo Finance failed for {symbol}: {str(e)}")
        
        return pd.DataFrame()
    
    def _get_web_scraped_data(self, symbol: str) -> pd.DataFrame:
        """Get data from web scraping public financial sites (FREE backup)"""
        try:
            if not self._check_rate_limit('web_scraping'):
                return pd.DataFrame()
            
            # Try to get data from public financial websites
            # This is a backup method when Yahoo Finance fails
            
            # For now, return empty DataFrame as web scraping requires more complex implementation
            # In a full implementation, you could scrape from sites like:
            # - MarketWatch
            # - Investing.com
            # - Yahoo Finance HTML (as backup)
            
            self.logger.info(f"Web scraping backup not fully implemented for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.warning(f"Web scraping failed for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_symbol_news_from_free_sources(self, symbol: str) -> Dict:
        """Get symbol-specific news from free public sources"""
        try:
            # Try to get news from Yahoo Finance (free)
            yahoo_news = self._get_yahoo_finance_news(symbol)
            if yahoo_news:
                return yahoo_news
            
            # If Yahoo Finance fails, try web scraping from public sites
            return self._scrape_news_from_public_sites(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting symbol news from free sources: {str(e)}")
            return {}
    
    def _get_general_news_from_free_sources(self) -> Dict:
        """Get general market news from free public sources"""
        try:
            # Try to get news from multiple free sources
            all_news = {}
            
            # Yahoo Finance news
            yahoo_news = self._get_yahoo_finance_general_news()
            if yahoo_news:
                all_news.update(yahoo_news)
            
            # Web scraped news from public sites
            scraped_news = self._scrape_general_news_from_public_sites()
            if scraped_news:
                all_news.update(scraped_news)
            
            return all_news
            
        except Exception as e:
            self.logger.error(f"Error getting general news from free sources: {str(e)}")
            return {}
    
    def _get_yahoo_finance_news(self, symbol: str) -> Dict:
        """Get news from Yahoo Finance (FREE)"""
        try:
            if not self._check_rate_limit('web_scraping'):
                return {}
            
            # Yahoo Finance news URL
            news_url = f"https://finance.yahoo.com/quote/{symbol}/news"
            
            response = self.session.get(news_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML for news headlines
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for news headlines (this is a simplified approach)
            headlines = []
            sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            # Find news elements (this would need to be adapted based on Yahoo's current HTML structure)
            news_elements = soup.find_all(['h3', 'h4', 'a'], class_=lambda x: x and 'news' in x.lower() if x else False)
            
            for element in news_elements[:10]:  # Limit to 10 headlines
                headline = element.get_text(strip=True)
                if headline and len(headline) > 10:
                    headlines.append(headline)
                    
                    # Analyze sentiment
                    sentiment = self._analyze_headline_sentiment(headline)
                    sentiment_counts[sentiment] += 1
            
            self._update_rate_limit('web_scraping')
            
            return {
                'bullish_news': sentiment_counts['bullish'],
                'bearish_news': sentiment_counts['bearish'],
                'neutral_news': sentiment_counts['neutral'],
                'key_headlines': headlines[:5],
                'data_source': 'free_yahoo_finance'
            }
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance news failed for {symbol}: {str(e)}")
            return {}
    
    def _get_yahoo_finance_general_news(self) -> Dict:
        """Get general market news from Yahoo Finance (FREE)"""
        try:
            if not self._check_rate_limit('web_scraping'):
                return {}
            
            # Yahoo Finance general news URL
            news_url = "https://finance.yahoo.com/news/"
            
            response = self.session.get(news_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML for news headlines
            soup = BeautifulSoup(response.content, 'html.parser')
            
            headlines = []
            sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
            
            # Find news elements (simplified approach)
            news_elements = soup.find_all(['h3', 'h4', 'a'], class_=lambda x: x and 'news' in x.lower() if x else False)
            
            for element in news_elements[:15]:  # Limit to 15 headlines
                headline = element.get_text(strip=True)
                if headline and len(headline) > 10:
                    headlines.append(headline)
                    
                    # Analyze sentiment
                    sentiment = self._analyze_headline_sentiment(headline)
                    sentiment_counts[sentiment] += 1
            
            self._update_rate_limit('web_scraping')
            
            return {
                'bullish_news': sentiment_counts['bullish'],
                'bearish_news': sentiment_counts['bearish'],
                'neutral_news': sentiment_counts['neutral'],
                'key_headlines': headlines[:5],
                'data_source': 'free_yahoo_finance'
            }
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance general news failed: {str(e)}")
            return {}
    
    def _scrape_news_from_public_sites(self, symbol: str) -> Dict:
        """Scrape news from public financial sites (FREE)"""
        try:
            if not self._check_rate_limit('web_scraping'):
                return {}
            
            # This is a placeholder for more sophisticated web scraping
            # In a full implementation, you could scrape from:
            # - Reuters
            # - Bloomberg
            # - MarketWatch
            # - Investing.com
            
            # For now, return empty to avoid being blocked
            return {}
            
        except Exception as e:
            self.logger.warning(f"News scraping failed for {symbol}: {str(e)}")
            return {}
    
    def _scrape_general_news_from_public_sites(self) -> Dict:
        """Scrape general news from public financial sites (FREE)"""
        try:
            if not self._check_rate_limit('web_scraping'):
                return {}
            
            # This is a placeholder for more sophisticated web scraping
            # In a full implementation, you could scrape from multiple public sites
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"General news scraping failed: {str(e)}")
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
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if free data source rate limit allows request"""
        try:
            if source_name not in self.last_request_time:
                return True
            
            limits = self.config.FREE_DATA_LIMITS.get(source_name, {})
            requests_per_minute = limits.get('requests_per_minute', 60)
            delay_between_requests = limits.get('delay_between_requests', 1)
            
            current_time = time.time()
            last_request = self.last_request_time[source_name]
            
            # Check if enough time has passed
            if current_time - last_request >= delay_between_requests:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit for {source_name}: {str(e)}")
            return True
    
    def _update_rate_limit(self, source_name: str):
        """Update free data source rate limit timestamp"""
        try:
            self.last_request_time[source_name] = time.time()
        except Exception as e:
            self.logger.error(f"Error updating rate limit for {source_name}: {str(e)}")
    
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