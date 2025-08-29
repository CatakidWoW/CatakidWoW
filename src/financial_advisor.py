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
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'BZ=F', 'HG=F', 'NG=F']
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
            
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions()
            
            # Get available instruments
            available_instruments = self._get_available_instruments()
            
            # Analyze each instrument with real-time data
            recommendations = []
            
            for instrument_type, symbols in available_instruments.items():
                for symbol in symbols[:10]:  # Limit to top 10 per type for performance
                    try:
                        # Get real-time market data
                        data = self.real_time_data.get_live_market_data(symbol, period='1d', interval='5m')
                        if data.empty:
                            continue
                        
                        # Get CFD setup analysis
                        cfd_setup = self.cfd_analyzer.analyze_cfd_setups(data, instrument_type)
                        if not cfd_setup:
                            continue
                        
                        # Get real-time news sentiment for this instrument
                        news_sentiment = self._get_instrument_news_sentiment(symbol)
                        
                        # Get real-time technical analysis
                        technical_analysis = self._analyze_instrument_technical(symbol, data)
                        
                        # Calculate exact entry prices from real data
                        entry_prices = self._calculate_exact_entry_prices(symbol, data, cfd_setup)
                        
                        # Check if trade is still executable with real-time data
                        trade_status = self._check_trade_executability(symbol, data, cfd_setup)
                        
                        # Calculate position sizing for £147 capital
                        position_sizing = self._calculate_position_sizing(capital, cfd_setup)
                        
                        # Create recommendation
                        recommendation = {
                            'symbol': symbol,
                            'instrument_type': instrument_type,
                            'setup_type': cfd_setup['setup_type'],
                            'direction': cfd_setup['direction'],
                            'confidence': cfd_setup['confidence'],
                            'setup_score': cfd_setup['setup_score'],
                            'entry_prices': entry_prices,
                            'stop_loss': cfd_setup['stop_loss'],
                            'take_profit': cfd_setup['take_profit'],
                            'news_sentiment': news_sentiment,
                            'technical_analysis': technical_analysis,
                            'trade_status': trade_status,
                            'position_sizing': position_sizing,
                            'risk_metrics': cfd_setup['risk_metrics'],
                            'market_conditions': market_conditions,
                            'recommendation_reason': self._generate_recommendation_reason(cfd_setup, news_sentiment, technical_analysis),
                            'execution_priority': self._calculate_execution_priority(cfd_setup, news_sentiment, technical_analysis),
                            'data_source': 'real_time',
                            'last_updated': datetime.now().isoformat()
                        }
                        
                        recommendations.append(recommendation)
                        
                    except Exception as e:
                        self.logger.warning(f"Error analyzing {symbol}: {str(e)}")
                        continue
            
            # Sort by execution priority and filter by criteria
            recommendations.sort(key=lambda x: x['execution_priority'], reverse=True)
            
            # Filter recommendations based on risk criteria
            filtered_recommendations = self._filter_recommendations(recommendations, capital)
            
            # Generate final analysis
            final_analysis = {
                'timestamp': datetime.now().isoformat(),
                'capital': capital,
                'market_conditions': market_conditions,
                'total_recommendations': len(recommendations),
                'filtered_recommendations': len(filtered_recommendations),
                'top_recommendations': filtered_recommendations[:5],  # Top 5 recommendations
                'risk_summary': self._generate_risk_summary(filtered_recommendations, capital),
                'execution_plan': self._generate_execution_plan(filtered_recommendations, capital),
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
    
    def _calculate_exact_entry_prices(self, symbol: str, data: pd.DataFrame, cfd_setup: Dict) -> Dict:
        """Calculate exact entry prices from real market data"""
        try:
            current_price = data['Close'].iloc[-1]
            direction = cfd_setup.get('direction', 'neutral')
            
            if direction == 'long':
                # For long positions, look for pullback entries
                entry_prices = {
                    'aggressive': round(current_price * 0.995, 4),  # 0.5% below current
                    'moderate': round(current_price * 0.99, 4),     # 1% below current
                    'conservative': round(current_price * 0.985, 4)  # 1.5% below current
                }
            elif direction == 'short':
                # For short positions, look for bounce entries
                entry_prices = {
                    'aggressive': round(current_price * 1.005, 4),  # 0.5% above current
                    'moderate': round(current_price * 1.01, 4),     # 1% above current
                    'conservative': round(current_price * 1.015, 4)  # 1.5% above current
                }
            else:
                # Neutral - use current price
                entry_prices = {
                    'current': round(current_price, 4)
                }
            
            # Add current market price and data source
            entry_prices['market_price'] = round(current_price, 4)
            entry_prices['data_source'] = 'real_time'
            entry_prices['last_updated'] = datetime.now().isoformat()
            
            return entry_prices
            
        except Exception as e:
            self.logger.error(f"Error calculating entry prices for {symbol}: {str(e)}")
            return {'market_price': round(data['Close'].iloc[-1], 4), 'data_source': 'real_time'}
    
    def _check_trade_executability(self, symbol: str, data: pd.DataFrame, cfd_setup: Dict) -> Dict:
        """Check if trade is still executable using real-time data"""
        try:
            current_price = data['Close'].iloc[-1]
            direction = cfd_setup.get('direction', 'neutral')
            
            # Check if price is still within reasonable range for entry
            if direction == 'long':
                # For long, check if price hasn't moved too high
                max_entry = current_price * 1.02  # 2% above current
                is_executable = True
                reason = "Price within long entry range"
            elif direction == 'short':
                # For short, check if price hasn't moved too low
                min_entry = current_price * 0.98  # 2% below current
                is_executable = True
                reason = "Price within short entry range"
            else:
                is_executable = False
                reason = "No clear direction"
            
            # Check if setup has expired (confidence too low)
            confidence = cfd_setup.get('confidence', 0)
            if confidence < 0.6:
                is_executable = False
                reason = f"Setup confidence too low: {confidence:.1%}"
            
            return {
                'is_executable': is_executable,
                'reason': reason,
                'current_price': current_price,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking trade executability for {symbol}: {str(e)}")
            return {'is_executable': False, 'reason': 'Error checking executability', 'data_source': 'real_time'}
    
    def _calculate_position_sizing(self, capital: float, cfd_setup: Dict) -> Dict:
        """Calculate position sizing for given capital"""
        try:
            risk_metrics = cfd_setup.get('risk_metrics', {})
            
            if not risk_metrics:
                return {
                    'position_size': 0,
                    'units': 0,
                    'risk_amount': 0,
                    'reason': 'No risk metrics available',
                    'data_source': 'real_time'
                }
            
            # Calculate position size based on 2% risk
            max_risk_amount = capital * self.risk_settings['max_risk_per_trade']
            risk_per_unit = risk_metrics.get('risk_amount', 0)
            
            if risk_per_unit > 0:
                units = max_risk_amount / risk_per_unit
                position_size = units * cfd_setup.get('entry_levels', {}).get('moderate', 0)
            else:
                units = 0
                position_size = 0
            
            return {
                'position_size': round(position_size, 2),
                'units': round(units, 2),
                'risk_amount': round(max_risk_amount, 2),
                'max_capital_usage': round((position_size / capital) * 100, 1),
                'reason': f"Based on {self.risk_settings['max_risk_per_trade']*100}% risk per trade",
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {str(e)}")
            return {'position_size': 0, 'units': 0, 'risk_amount': 0, 'data_source': 'real_time'}
    
    def _generate_recommendation_reason(self, cfd_setup: Dict, news_sentiment: Dict, technical_analysis: Dict) -> str:
        """Generate human-readable recommendation reason from real data"""
        try:
            reasons = []
            
            # CFD setup reasons
            setup_type = cfd_setup.get('setup_type', 'Unknown')
            confidence = cfd_setup.get('confidence', 0)
            direction = cfd_setup.get('direction', 'neutral')
            
            reasons.append(f"CFD Setup: {setup_type} with {confidence:.1%} confidence")
            reasons.append(f"Direction: {direction.upper()} position recommended")
            
            # News sentiment reasons
            news_sent = news_sentiment.get('sentiment', 'NEUTRAL')
            if news_sent != 'NEUTRAL':
                reasons.append(f"News Sentiment: {news_sent.lower()} market sentiment")
            
            # Technical analysis reasons
            tech_score = technical_analysis.get('technical_score', 0.5)
            if tech_score > 0.7:
                reasons.append("Technical Analysis: Strong technical indicators")
            elif tech_score < 0.3:
                reasons.append("Technical Analysis: Weak technical indicators")
            
            # Risk/reward reasons
            risk_metrics = cfd_setup.get('risk_metrics', {})
            if risk_metrics:
                rr_ratio = risk_metrics.get('risk_reward_ratio', 0)
                if rr_ratio >= 2.0:
                    reasons.append(f"Risk/Reward: Excellent {rr_ratio:.1f}:1 ratio")
                elif rr_ratio >= 1.5:
                    reasons.append(f"Risk/Reward: Good {rr_ratio:.1f}:1 ratio")
            
            # Data source
            reasons.append("Data Source: Real-time market data")
            
            return " | ".join(reasons)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation reason: {str(e)}")
            return "CFD setup analysis completed with real-time data"
    
    def _calculate_execution_priority(self, cfd_setup: Dict, news_sentiment: Dict, technical_analysis: Dict) -> float:
        """Calculate execution priority score from real data"""
        try:
            priority = 0.0
            
            # CFD setup priority (40% weight)
            confidence = cfd_setup.get('confidence', 0)
            priority += confidence * 0.4
            
            # News sentiment priority (30% weight)
            news_sent = news_sentiment.get('sentiment', 'NEUTRAL')
            if news_sent == 'BULLISH':
                priority += 0.3
            elif news_sent == 'BEARISH':
                priority += 0.2
            else:
                priority += 0.1
            
            # Technical analysis priority (30% weight)
            tech_score = technical_analysis.get('technical_score', 0.5)
            priority += tech_score * 0.3
            
            return priority
            
        except Exception as e:
            self.logger.error(f"Error calculating execution priority: {str(e)}")
            return 0.5
    
    def _filter_recommendations(self, recommendations: List[Dict], capital: float) -> List[Dict]:
        """Filter recommendations based on risk criteria"""
        try:
            filtered = []
            
            for rec in recommendations:
                # Check if trade is executable
                if not rec.get('trade_status', {}).get('is_executable', False):
                    continue
                
                # Check confidence threshold
                if rec.get('confidence', 0) < 0.6:
                    continue
                
                # Check risk/reward ratio
                risk_metrics = rec.get('risk_metrics', {})
                if risk_metrics:
                    rr_ratio = risk_metrics.get('risk_reward_ratio', 0)
                    if rr_ratio < self.risk_settings['min_risk_reward']:
                        continue
                
                # Check position sizing
                position_sizing = rec.get('position_sizing', {})
                if position_sizing.get('position_size', 0) > capital * 0.5:  # Max 50% of capital per trade
                    continue
                
                filtered.append(rec)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error filtering recommendations: {str(e)}")
            return recommendations
    
    def _generate_risk_summary(self, recommendations: List[Dict], capital: float) -> Dict:
        """Generate risk summary for all recommendations"""
        try:
            if not recommendations:
                return {'total_risk': 0, 'max_drawdown': 0, 'risk_level': 'LOW', 'data_source': 'real_time'}
            
            total_risk = sum(rec.get('position_sizing', {}).get('risk_amount', 0) for rec in recommendations)
            max_risk_per_trade = max(rec.get('position_sizing', {}).get('risk_amount', 0) for rec in recommendations)
            
            # Calculate risk level
            risk_percentage = (total_risk / capital) * 100
            if risk_percentage > 10:
                risk_level = 'HIGH'
            elif risk_percentage > 5:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'total_risk': round(total_risk, 2),
                'max_risk_per_trade': round(max_risk_per_trade, 2),
                'total_risk_percentage': round(risk_percentage, 1),
                'risk_level': risk_level,
                'capital': capital,
                'recommendations_count': len(recommendations),
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {str(e)}")
            return {'total_risk': 0, 'risk_level': 'UNKNOWN', 'data_source': 'real_time'}
    
    def _generate_execution_plan(self, recommendations: List[Dict], capital: float) -> Dict:
        """Generate execution plan for recommendations"""
        try:
            if not recommendations:
                return {'execution_steps': [], 'total_capital_required': 0, 'data_source': 'real_time'}
            
            execution_steps = []
            total_capital_required = 0
            
            for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
                symbol = rec.get('symbol', 'Unknown')
                direction = rec.get('direction', 'neutral')
                entry_prices = rec.get('entry_prices', {})
                position_sizing = rec.get('position_sizing', {})
                
                # Determine entry price
                if direction == 'long':
                    entry_price = entry_prices.get('moderate', entry_prices.get('market_price', 0))
                    entry_type = 'Buy at pullback'
                elif direction == 'short':
                    entry_price = entry_prices.get('moderate', entry_prices.get('market_price', 0))
                    entry_type = 'Sell at bounce'
                else:
                    continue
                
                # Calculate capital required
                units = position_sizing.get('units', 0)
                capital_required = units * entry_price
                total_capital_required += capital_required
                
                execution_step = {
                    'step': i,
                    'symbol': symbol,
                    'action': f"{direction.upper()} {entry_type}",
                    'entry_price': round(entry_price, 4),
                    'units': round(units, 2),
                    'capital_required': round(capital_required, 2),
                    'stop_loss': rec.get('stop_loss', {}).get('level'),
                    'take_profit': rec.get('take_profit', {}).get('level'),
                    'priority': 'HIGH' if i == 1 else 'MEDIUM' if i == 2 else 'LOW',
                    'data_source': 'real_time'
                }
                
                execution_steps.append(execution_step)
            
            return {
                'execution_steps': execution_steps,
                'total_capital_required': round(total_capital_required, 2),
                'capital_available': capital,
                'execution_notes': [
                    "Execute trades in priority order",
                    "Set stop losses immediately after entry",
                    "Monitor positions throughout the day",
                    "Exit positions before market close for same-day trading",
                    "All data is real-time and live"
                ],
                'data_source': 'real_time'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating execution plan: {str(e)}")
            return {'execution_steps': [], 'total_capital_required': 0, 'data_source': 'real_time'}
    
    def _get_volatility_trading_implication(self, volatility_condition: str) -> str:
        """Get trading implications based on volatility"""
        implications = {
            'HIGH': 'Use wider stop losses, reduce position sizes, focus on momentum trades',
            'LOW': 'Tight stop losses, normal position sizes, range-bound trading strategies',
            'NORMAL': 'Standard risk management, balanced position sizing'
        }
        return implications.get(volatility_condition, 'Standard risk management')