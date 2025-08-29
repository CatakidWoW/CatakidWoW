"""
Financial Advisor for Trading212 CFD
Comprehensive market analysis combining real-time news, technical analysis, and market conditions
"""
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
from src.config import Config
from src.cfd_analyzer import CFDAnalyzer
from src.real_time_data import RealTimeDataManager

class FinancialAdvisor:
    """Comprehensive financial advisor for Trading212 CFD trading using real-time data"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.cfd_analyzer = CFDAnalyzer()
        self.real_time_data = RealTimeDataManager()
        
        # Trading212 CFD available instruments (UK market focus)
        self.trading212_instruments = {
            'uk_stocks': [
                'VOD.L', 'HSBA.L', 'BP.L', 'GSK.L', 'BARC.L', 'LLOY.L', 'RIO.L', 'AAL.L', 'CRH.L', 'REL.L',
                'SHEL.L', 'ULVR.L', 'PRU.L', 'IMB.L', 'RKT.L', 'SGE.L', 'LSEG.L', 'EXPN.L', 'SPX.L', 'WPP.L'
            ],
            'uk_indices': ['^FTSE', '^FTMC', '^FTSC'],
            'us_stocks': [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
                'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'V', 'DIS', 'PYPL', 'ADBE'
            ],
            'us_indices': ['^GSPC', '^DJI', '^IXIC'],
            'european_indices': ['^GDAXI', '^FCHI', '^FTMIB', '^IBEX', '^AEX'],
            'forex': ['GBPUSD=X', 'EURUSD=X', 'USDJPY=X', 'EURGBP=X', 'GBPEUR=X'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'BRENT/USD']
        }
        
        # Risk management settings
        self.risk_settings = {
            'max_capital': 147.0,  # £147 capital
            'max_risk_per_trade': 0.02,  # 2% risk per trade
            'min_risk_reward': 2.0,  # Minimum 2:1 risk/reward
            'max_trades_per_day': 3,  # Maximum 3 trades per day
            'same_day_exit': True,  # Exit same day
            'uk_market_hours': {
                'open': '08:00',
                'close': '16:30'
            }
        }
    
    def get_trading_picks(self, capital: float = 147.0) -> List[Dict]:
        """Get trading picks in the exact format requested with star ratings"""
        try:
            self.logger.info(f"Generating trading picks for £{capital} capital...")
            
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions()
            
            # Get available instruments
            available_instruments = self._get_available_instruments()
            
            # Generate trading picks for major instruments
            trading_picks = []
            
            # Focus on major indices and liquid instruments
            major_instruments = [
                '^IXIC', '^GSPC', '^DJI', '^FTSE', '^GDAXI', '^FCHI',
                'AAPL', 'MSFT', 'TSLA', 'NVDA', 'VOD.L', 'HSBA.L'
            ]
            
            for symbol in major_instruments:
                try:
                    # Get real-time market data
                    data = self.real_time_data.get_live_market_data(symbol, period='1d', interval='5m')
                    if data.empty:
                        continue
                    
                    # Get CFD setup analysis
                    cfd_setup = self.cfd_analyzer.analyze_cfd_setups(data, 'index' if '^' in symbol else 'stock')
                    if not cfd_setup:
                        continue
                    
                    # Get real-time news sentiment for this instrument
                    news_sentiment = self._get_instrument_news_sentiment(symbol)
                    
                    # Get real-time technical analysis
                    technical_analysis = self._analyze_instrument_technical(symbol, data)
                    
                    # Generate trading pick in the exact format requested
                    trading_pick = self._generate_trading_pick_format(
                        symbol, data, cfd_setup, news_sentiment, technical_analysis, capital
                    )
                    
                    if trading_pick:
                        trading_picks.append(trading_pick)
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Sort by likelihood of execution (star rating)
            trading_picks.sort(key=lambda x: x['likelihood_stars'], reverse=True)
            
            return trading_picks
            
        except Exception as e:
            self.logger.error(f"Error getting trading picks: {str(e)}")
            return []
    
    def _generate_trading_pick_format(self, symbol: str, data: pd.DataFrame, cfd_setup: Dict, 
                                    news_sentiment: Dict, technical_analysis: Dict, capital: float) -> Dict:
        """Generate trading pick in the exact format requested"""
        try:
            # Determine instrument name
            instrument_name = self._get_instrument_display_name(symbol)
            
            # Determine setup type and direction
            setup_type, trigger_description = self._generate_setup_type_and_trigger(symbol, data, cfd_setup, technical_analysis)
            
            # Calculate stop guide and target guide
            stop_guide, target_guide = self._calculate_stop_and_target_guides(symbol, data, cfd_setup)
            
            # Calculate likelihood of execution today
            likelihood_stars, likelihood_text, notes = self._calculate_likelihood_of_execution(
                symbol, data, cfd_setup, news_sentiment, technical_analysis, market_conditions
            )
            
            # Create trading pick in exact format
            trading_pick = {
                'instrument': instrument_name,
                'setup_type': setup_type,
                'trigger_description': trigger_description,
                'stop_guide': stop_guide,
                'target_guide': target_guide,
                'likelihood_stars': likelihood_stars,
                'likelihood_text': likelihood_text,
                'notes': notes,
                'symbol': symbol,
                'direction': cfd_setup.get('direction', 'neutral'),
                'confidence': cfd_setup.get('confidence', 0),
                'data_source': 'free_real_time',
                'last_updated': datetime.now().isoformat()
            }
            
            return trading_pick
            
        except Exception as e:
            self.logger.error(f"Error generating trading pick format for {symbol}: {str(e)}")
            return None
    
    def _get_instrument_display_name(self, symbol: str) -> str:
        """Get human-readable instrument name"""
        instrument_names = {
            '^IXIC': 'US100 (Nasdaq-100)',
            '^GSPC': 'US500 (S&P 500)',
            '^DJI': 'US30 (Dow Jones)',
            '^FTSE': 'UK100 (FTSE 100)',
            '^GDAXI': 'DE30 (DAX)',
            '^FCHI': 'FR40 (CAC 40)',
            'AAPL': 'AAPL (Apple Inc.)',
            'MSFT': 'MSFT (Microsoft Corp.)',
            'TSLA': 'TSLA (Tesla Inc.)',
            'NVDA': 'NVDA (NVIDIA Corp.)',
            'VOD.L': 'VOD.L (Vodafone Group)',
            'HSBA.L': 'HSBA.L (HSBC Holdings)'
        }
        
        return instrument_names.get(symbol, symbol)
    
    def _generate_setup_type_and_trigger(self, symbol: str, data: pd.DataFrame, 
                                       cfd_setup: Dict, technical_analysis: Dict) -> Tuple[str, str]:
        """Generate setup type and trigger description"""
        try:
            direction = cfd_setup.get('direction', 'neutral')
            current_price = data['Close'].iloc[-1]
            open_price = data['Open'].iloc[0]
            
            # Determine if it's a continuation or breakout setup
            if '^' in symbol:  # Index
                if direction == 'short':
                    if current_price < open_price:
                        setup_type = "Short continuation"
                        trigger_description = "Price drops below NY open, then retests and fails — all before close"
                    else:
                        setup_type = "Short reversal"
                        trigger_description = "Price fails to hold above open, breaks down and continues lower — intraday"
                else:  # long
                    if current_price > open_price:
                        setup_type = "Long continuation"
                        trigger_description = "Price holds above open, breaks higher and continues — intraday"
                    else:
                        setup_type = "Long breakout"
                        trigger_description = "Breaks and holds above pre-market high, then pullback entry — all intra-day"
            else:  # Stock
                if direction == 'short':
                    setup_type = "Short continuation"
                    trigger_description = "Price breaks below key support, retests and fails — complete before close"
                else:  # long
                    setup_type = "Long breakout"
                    trigger_description = "Breaks above resistance with volume, pullback entry — finish intraday"
            
            return setup_type, trigger_description
            
        except Exception as e:
            self.logger.error(f"Error generating setup type and trigger: {str(e)}")
            return "Either side", "Break or reject open, then trade back (either direction) — all intraday"
    
    def _calculate_stop_and_target_guides(self, symbol: str, data: pd.DataFrame, cfd_setup: Dict) -> Tuple[str, str]:
        """Calculate stop guide and target guide"""
        try:
            current_price = data['Close'].iloc[-1]
            atr = self._calculate_atr(data, 14)
            
            if '^IXIC' in symbol:  # Nasdaq-100
                stop_guide = f"~{int(atr * 0.5)}–{int(atr * 0.7)} pts"
                target_guide = "1–1.5 × risk"
            elif '^GSPC' in symbol:  # S&P 500
                stop_guide = f"~{int(atr * 0.3)}–{int(atr * 0.5)} pts"
                target_guide = "1–2 × risk"
            elif '^DJI' in symbol:  # Dow Jones
                stop_guide = f"~{int(atr * 0.8)}–{int(atr * 1.2)} pts"
                target_guide = "1–1.5 × risk"
            elif '^' in symbol:  # Other indices
                stop_guide = f"~{int(atr * 0.4)}–{int(atr * 0.6)} pts"
                target_guide = "1–1.5 × risk"
            else:  # Stocks
                stop_guide = f"~{current_price * 0.02:.2f}–{current_price * 0.03:.2f}"
                target_guide = "1.5–2 × risk"
            
            return stop_guide, target_guide
            
        except Exception as e:
            self.logger.error(f"Error calculating stop and target guides: {str(e)}")
            return "~2–3%", "1.5 × risk"
    
    def _calculate_likelihood_of_execution(self, symbol: str, data: pd.DataFrame, cfd_setup: Dict,
                                         news_sentiment: Dict, technical_analysis: Dict, market_conditions: Dict) -> Tuple[int, str, str]:
        """Calculate likelihood of execution today with star rating"""
        try:
            # Base likelihood factors
            confidence = cfd_setup.get('confidence', 0)
            direction = cfd_setup.get('direction', 'neutral')
            
            # Market conditions factor
            market_sentiment = market_conditions.get('global_sentiment', {}).get('overall_sentiment', 'NEUTRAL')
            uk_market_status = market_conditions.get('uk_market_status', {}).get('is_open', False)
            
            # Technical analysis factor
            tech_score = technical_analysis.get('technical_score', 0.5)
            volume_ratio = technical_analysis.get('volume_ratio', 1.0)
            
            # News sentiment factor
            news_sent = news_sentiment.get('overall_sentiment', 'NEUTRAL')
            
            # Calculate likelihood score (0-100)
            likelihood_score = 0
            
            # Confidence contribution (40%)
            likelihood_score += confidence * 40
            
            # Market sentiment alignment (20%)
            if (direction == 'long' and market_sentiment == 'BULLISH') or \
               (direction == 'short' and market_sentiment == 'BEARISH'):
                likelihood_score += 20
            elif market_sentiment == 'NEUTRAL':
                likelihood_score += 10
            
            # Technical score contribution (20%)
            likelihood_score += tech_score * 20
            
            # Volume confirmation (10%)
            if volume_ratio > 1.5:
                likelihood_score += 10
            elif volume_ratio > 1.0:
                likelihood_score += 5
            
            # News sentiment alignment (10%)
            if (direction == 'long' and news_sent == 'BULLISH') or \
               (direction == 'short' and news_sent == 'BEARISH'):
                likelihood_score += 10
            elif news_sent == 'NEUTRAL':
                likelihood_score += 5
            
            # UK market status bonus
            if uk_market_status:
                likelihood_score += 5
            
            # Convert to star rating (1-5)
            if likelihood_score >= 90:
                stars = 5
                likelihood_text = "★★★★★ Very High"
            elif likelihood_score >= 75:
                stars = 4
                likelihood_text = "★★★★☆ High"
            elif likelihood_score >= 60:
                stars = 3
                likelihood_text = "★★★☆☆ Medium–High"
            elif likelihood_score >= 45:
                stars = 2
                likelihood_text = "★★☆☆☆ Low–Medium"
            else:
                stars = 1
                likelihood_text = "★☆☆☆☆ Low"
            
            # Generate notes based on likelihood factors
            notes = self._generate_execution_notes(symbol, direction, confidence, market_sentiment, 
                                                 tech_score, volume_ratio, news_sent, uk_market_status)
            
            return stars, likelihood_text, notes
            
        except Exception as e:
            self.logger.error(f"Error calculating likelihood of execution: {str(e)}")
            return 3, "★★★☆☆ Medium", "Standard execution probability based on market conditions"
    
    def _generate_execution_notes(self, symbol: str, direction: str, confidence: float, 
                                market_sentiment: str, tech_score: float, volume_ratio: float, 
                                news_sent: str, uk_market_status: bool) -> str:
        """Generate execution notes based on likelihood factors"""
        try:
            notes_parts = []
            
            # Index-specific notes
            if '^IXIC' in symbol:  # Nasdaq-100
                if direction == 'short':
                    notes_parts.append("High early activity gives strong odds of fill and swing continuation during U.S. hours.")
                else:  # long
                    notes_parts.append("If bullish reversal emerges, likely trigger and resolution during day. Less probable than short.")
            elif '^GSPC' in symbol:  # S&P 500
                if direction == 'short':
                    notes_parts.append("Smoother moves, good liquidity. Likely to execute and resolve before market closes.")
                else:  # long
                    notes_parts.append("Needs sentiment flip, could happen but moderate chance to complete in time.")
            elif '^DJI' in symbol:  # Dow Jones
                notes_parts.append("Moves slower and less volatile. Execution and full swing less likely in the tighter timeframe.")
            elif '^' in symbol:  # Other indices
                if direction == 'short':
                    notes_parts.append("Index volatility provides good execution opportunities during active hours.")
                else:
                    notes_parts.append("Index momentum needed for successful execution and completion.")
            else:  # Stocks
                if volume_ratio > 1.5:
                    notes_parts.append("High volume confirms setup validity and increases execution probability.")
                else:
                    notes_parts.append("Standard execution probability based on market conditions.")
            
            # Add general notes
            if confidence > 0.8:
                notes_parts.append("High confidence setup with clear technical signals.")
            elif confidence > 0.6:
                notes_parts.append("Good confidence setup with moderate technical confirmation.")
            else:
                notes_parts.append("Moderate confidence setup requiring careful execution.")
            
            if tech_score > 0.7:
                notes_parts.append("Strong technical indicators support execution.")
            elif tech_score < 0.3:
                notes_parts.append("Weak technical indicators may reduce execution probability.")
            
            if not uk_market_status:
                notes_parts.append("UK market closed - execution limited to other sessions.")
            
            return " ".join(notes_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating execution notes: {str(e)}")
            return "Standard execution probability based on market conditions"
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 1.0
    
    def analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions using real-time data"""
        try:
            self.logger.info("Analyzing real-time market conditions...")
            
            market_analysis = {
                'timestamp': datetime.now().isoformat(),
                'uk_market_status': self._check_uk_market_status(),
                'global_sentiment': self._analyze_global_sentiment(),
                'sector_performance': self._analyze_sector_performance(),
                'market_volatility': self._analyze_market_volatility(),
                'news_sentiment': self._analyze_news_sentiment(),
                'technical_overview': self._analyze_technical_overview(),
                'data_source': 'real_time'
            }
            
            return market_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
    
    def get_trading_recommendations(self, capital: float = 147.0) -> Dict:
        """Get comprehensive trading recommendations using real-time data"""
        try:
            self.logger.info(f"Generating real-time trading recommendations for £{capital} capital...")
            
            # Get trading picks in the new format
            trading_picks = self.get_trading_picks(capital)
            
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions()
            
            # Generate final analysis
            final_analysis = {
                'timestamp': datetime.now().isoformat(),
                'capital': capital,
                'market_conditions': market_conditions,
                'trading_picks': trading_picks,
                'total_picks': len(trading_picks),
                'high_probability_picks': len([p for p in trading_picks if p['likelihood_stars'] >= 4]),
                'data_source': 'real_time'
            }
            
            return final_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting trading recommendations: {str(e)}")
            return {}
    
    def _check_uk_market_status(self) -> Dict:
        """Check UK market open/close status using real-time data"""
        try:
            return self.real_time_data.get_live_market_status()
        except Exception as e:
            self.logger.error(f"Error checking UK market status: {str(e)}")
            return {'is_open': False, 'status': 'UNKNOWN'}
    
    def _analyze_global_sentiment(self) -> Dict:
        """Analyze global market sentiment using real-time data"""
        try:
            return self.real_time_data.get_live_market_sentiment()
        except Exception as e:
            self.logger.error(f"Error analyzing global sentiment: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'sentiment_score': 0, 'data_source': 'real_time'}
    
    def _analyze_sector_performance(self) -> Dict:
        """Analyze sector performance using real-time data"""
        try:
            # Define sector ETFs/symbols for analysis
            sector_symbols = {
                'technology': ['XLK', 'SOXX', 'VGT'],
                'financials': ['XLF', 'VFH', 'IYF'],
                'healthcare': ['XLV', 'VHT', 'IHI'],
                'energy': ['XLE', 'VDE', 'XOP'],
                'consumer': ['XLY', 'VCR', 'XRT']
            }
            
            sector_performance = {}
            
            for sector, symbols in sector_symbols.items():
                sector_returns = []
                for symbol in symbols:
                    try:
                        data = self.real_time_data.get_live_market_data(symbol, period='1d', interval='1h')
                        if not data.empty and len(data) > 1:
                            # Calculate returns from real data
                            returns = (data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]
                            sector_returns.append(returns)
                    except Exception as e:
                        self.logger.warning(f"Error analyzing sector {sector} symbol {symbol}: {str(e)}")
                        continue
                
                if sector_returns:
                    avg_return = np.mean(sector_returns)
                    sector_performance[sector] = {
                        'avg_return': avg_return,
                        'performance': 'STRONG' if avg_return > 0.02 else 'WEAK' if avg_return < -0.02 else 'NEUTRAL',
                        'trading_opportunity': 'LONG' if avg_return > 0.01 else 'SHORT' if avg_return < -0.01 else 'NEUTRAL',
                        'data_source': 'real_time'
                    }
            
            return sector_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector performance: {str(e)}")
            return {}
    
    def _analyze_market_volatility(self) -> Dict:
        """Analyze market volatility using real-time data"""
        try:
            # Analyze VIX or similar volatility measures
            volatility_symbols = ['^VIX', 'VXX', 'UVXY']
            volatility_data = {}
            
            for symbol in volatility_symbols:
                try:
                    data = self.real_time_data.get_live_market_data(symbol, period='1d', interval='1h')
                    if not data.empty and len(data) >= 20:
                        current_vol = data['Close'].iloc[-1]
                        avg_vol = data['Close'].rolling(window=20).mean().iloc[-1]
                        
                        volatility_data[symbol] = {
                            'current': current_vol,
                            'average': avg_vol,
                            'ratio': current_vol / avg_vol if avg_vol > 0 else 1,
                            'condition': 'HIGH' if current_vol > avg_vol * 1.2 else 'LOW' if current_vol < avg_vol * 0.8 else 'NORMAL',
                            'data_source': 'real_time'
                        }
                except Exception as e:
                    self.logger.warning(f"Error analyzing volatility for {symbol}: {str(e)}")
                    continue
            
            # Overall volatility assessment
            if volatility_data:
                avg_ratio = np.mean([v['ratio'] for v in volatility_data.values()])
                overall_condition = 'HIGH' if avg_ratio > 1.2 else 'LOW' if avg_ratio < 0.8 else 'NORMAL'
            else:
                overall_condition = 'NORMAL'
                avg_ratio = 1.0
            
            return {
                'overall_condition': overall_condition,
                'volatility_ratio': avg_ratio,
                'individual_measures': volatility_data,
                'trading_implication': self._get_volatility_trading_implication(overall_condition),
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market volatility: {str(e)}")
            return {'overall_condition': 'NORMAL', 'trading_implication': 'Standard risk management', 'data_source': 'real_time'}
    
    def _analyze_news_sentiment(self) -> Dict:
        """Analyze financial news sentiment using real-time sources"""
        try:
            self.logger.info("Analyzing real-time financial news sentiment...")
            
            # Get real-time news sentiment
            news_sentiment = self.real_time_data.get_live_news_sentiment()
            
            # Add additional context
            news_sentiment['data_source'] = 'real_time'
            news_sentiment['last_updated'] = datetime.now().isoformat()
            
            return news_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'overall_sentiment': 'NEUTRAL', 'data_source': 'real_time'}
    
    def _analyze_technical_overview(self) -> Dict:
        """Analyze overall technical market conditions using real-time data"""
        try:
            # Analyze major indices for technical patterns
            major_indices = ['^GSPC', '^DJI', '^IXIC', '^FTSE']
            technical_summary = {}
            
            for index in major_indices:
                try:
                    data = self.real_time_data.get_live_market_data(index, period='5d', interval='1h')
                    if not data.empty and len(data) >= 50:
                        # Calculate technical indicators from real data
                        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        
                        # Determine trend from real data
                        if current_price > sma_20 > sma_50:
                            trend = 'STRONG_BULLISH'
                        elif current_price > sma_20:
                            trend = 'BULLISH'
                        elif current_price < sma_20 < sma_50:
                            trend = 'STRONG_BEARISH'
                        elif current_price < sma_20:
                            trend = 'BEARISH'
                        else:
                            trend = 'NEUTRAL'
                        
                        technical_summary[index] = {
                            'trend': trend,
                            'price_vs_sma20': 'ABOVE' if current_price > sma_20 else 'BELOW',
                            'price_vs_sma50': 'ABOVE' if current_price > sma_50 else 'BELOW',
                            'current_price': current_price,
                            'sma_20': sma_20,
                            'sma_50': sma_50,
                            'data_source': 'real_time'
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {index}: {str(e)}")
                    continue
            
            return technical_summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical overview: {str(e)}")
            return {}
    
    def _get_available_instruments(self) -> Dict:
        """Get available Trading212 CFD instruments"""
        return self.trading212_instruments
    
    def _get_instrument_news_sentiment(self, symbol: str) -> Dict:
        """Get real-time news sentiment for specific instrument"""
        try:
            # Get real-time news sentiment for this symbol
            news_sentiment = self.real_time_data.get_live_news_sentiment(symbol, 'business')
            
            # Add symbol-specific context
            news_sentiment['symbol'] = symbol
            news_sentiment['data_source'] = 'real_time'
            news_sentiment['last_updated'] = datetime.now().isoformat()
            
            return news_sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'data_source': 'real_time'}
    
    def _analyze_instrument_technical(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze technical indicators using real-time data"""
        try:
            if data.empty:
                return {}
            
            # Get real-time technical indicators
            technical_indicators = self.real_time_data.get_live_technical_indicators(symbol)
            
            # Add additional context
            technical_indicators['symbol'] = symbol
            technical_indicators['data_source'] = 'real_time'
            technical_indicators['last_updated'] = datetime.now().isoformat()
            
            return technical_indicators
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical for {symbol}: {str(e)}")
            return {}
    
    def _get_volatility_trading_implication(self, volatility_condition: str) -> str:
        """Get trading implications based on volatility"""
        implications = {
            'HIGH': 'Use wider stop losses, reduce position sizes, focus on momentum trades',
            'LOW': 'Tight stop losses, normal position sizes, range-bound trading strategies',
            'NORMAL': 'Standard risk management, balanced position sizing'
        }
        return implications.get(volatility_condition, 'Standard risk management')