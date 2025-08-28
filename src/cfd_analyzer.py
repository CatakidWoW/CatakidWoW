"""
CFD Analyzer for Trading212
Specialized analyzer for finding the best CFD trading setups
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from src.config import Config

class CFDAnalyzer:
    """Specialized CFD analyzer for Trading212 trading setups"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # CFD-specific settings
        self.cfd_instruments = {
            'stocks': {
                'US': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'],
                'UK': ['VOD.L', 'HSBA.L', 'BP.L', 'GSK.L', 'BARC.L', 'LLOY.L', 'RIO.L', 'AAL.L', 'CRH.L', 'REL.L'],
                'EU': ['ASML.AS', 'SAP.DE', 'NESN.SW', 'NOVO.CO', 'ASML.AS', 'SAP.DE', 'NESN.SW', 'NOVO.CO'],
                'AU': ['CBA.AX', 'CSL.AX', 'BHP.AX', 'RIO.AX', 'WBC.AX', 'ANZ.AX', 'NAB.AX', 'MQG.AX']
            },
            'indices': ['US500', 'US30', 'US100', 'UK100', 'DE30', 'FR40', 'IT40', 'ES35', 'AU200', 'JP225'],
            'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP'],
            'commodities': ['XAU/USD', 'XAG/USD', 'WTI/USD', 'BRENT/USD', 'COPPER/USD', 'NATURAL_GAS/USD'],
            'crypto': ['BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD', 'LTC/USD', 'BCH/USD']
        }
        
        # CFD setup criteria
        self.setup_criteria = {
            'trend_strength': 0.7,      # Minimum trend strength (0-1)
            'volatility_threshold': 0.02, # Minimum volatility for opportunities
            'volume_ratio': 1.5,         # Minimum volume ratio
            'risk_reward_ratio': 2.0,    # Minimum risk/reward ratio
            'setup_confidence': 0.6      # Minimum confidence score
        }
    
    def analyze_cfd_setups(self, data: pd.DataFrame, instrument_type: str = 'stock') -> Dict:
        """
        Analyze CFD setups for a given instrument
        
        Args:
            data: OHLCV data with technical indicators
            instrument_type: Type of CFD instrument
        
        Returns:
            Dictionary with CFD setup analysis
        """
        if data.empty:
            return {}
        
        try:
            # Calculate setup scores
            trend_score = self._calculate_trend_score(data)
            momentum_score = self._calculate_momentum_score(data)
            volatility_score = self._calculate_volatility_score(data)
            volume_score = self._calculate_volume_score(data)
            support_resistance_score = self._calculate_support_resistance_score(data)
            
            # Calculate overall setup score
            setup_score = (
                trend_score * 0.3 +
                momentum_score * 0.25 +
                volatility_score * 0.2 +
                volume_score * 0.15 +
                support_resistance_score * 0.1
            )
            
            # Determine setup type and direction
            setup_type, direction, confidence = self._determine_setup_type(data, setup_score)
            
            # Calculate entry, stop loss, and take profit levels
            entry_levels = self._calculate_entry_levels(data, direction)
            stop_loss = self._calculate_stop_loss(data, direction, entry_levels)
            take_profit = self._calculate_take_profit(data, direction, entry_levels, stop_loss)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(data, entry_levels, stop_loss, take_profit)
            
            # Generate setup summary
            setup_summary = {
                'instrument_type': instrument_type,
                'setup_type': setup_type,
                'direction': direction,
                'confidence': confidence,
                'setup_score': setup_score,
                'entry_levels': entry_levels,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_metrics': risk_metrics,
                'trend_analysis': {
                    'trend_score': trend_score,
                    'trend_strength': self._get_trend_strength(data),
                    'trend_direction': self._get_trend_direction(data)
                },
                'momentum_analysis': {
                    'momentum_score': momentum_score,
                    'rsi': data['RSI'].iloc[-1] if 'RSI' in data.columns else None,
                    'macd_signal': self._get_macd_signal(data)
                },
                'volatility_analysis': {
                    'volatility_score': volatility_score,
                    'current_volatility': data['Volatility'].iloc[-1] if 'Volatility' in data.columns else None,
                    'volatility_trend': self._get_volatility_trend(data)
                },
                'volume_analysis': {
                    'volume_score': volume_score,
                    'volume_ratio': data['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in data.columns else None,
                    'volume_trend': self._get_volume_trend(data)
                },
                'support_resistance': {
                    'support_levels': self._find_support_levels(data),
                    'resistance_levels': self._find_resistance_levels(data),
                    'current_position': self._get_price_position(data)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return setup_summary
            
        except Exception as e:
            self.logger.error(f"Error analyzing CFD setup: {str(e)}")
            return {}
    
    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """Calculate trend strength score"""
        try:
            # Use multiple timeframe analysis
            short_ma = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else data['Close'].iloc[-1]
            long_ma = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else data['Close'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Calculate trend strength
            ma_trend = 1.0 if short_ma > long_ma else 0.0
            price_vs_ma = 1.0 if current_price > short_ma else 0.0
            
            # Calculate slope of moving averages
            if len(data) >= 20:
                ma_slope = (data['SMA_20'].iloc[-1] - data['SMA_20'].iloc[-10]) / data['SMA_20'].iloc[-10]
                slope_score = min(max(ma_slope * 10, 0), 1)  # Normalize to 0-1
            else:
                slope_score = 0.5
            
            trend_score = (ma_trend * 0.4 + price_vs_ma * 0.3 + slope_score * 0.3)
            return min(max(trend_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend score: {str(e)}")
            return 0.5
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            momentum_score = 0.5
            
            # RSI analysis
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if rsi < 30:  # Oversold
                    momentum_score += 0.3
                elif rsi > 70:  # Overbought
                    momentum_score -= 0.3
                elif 40 <= rsi <= 60:  # Neutral
                    momentum_score += 0.1
            
            # MACD analysis
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_Signal'].iloc[-1]
                macd_histogram = data['MACD_Histogram'].iloc[-1]
                
                if macd > macd_signal and macd_histogram > 0:
                    momentum_score += 0.2
                elif macd < macd_signal and macd_histogram < 0:
                    momentum_score -= 0.2
            
            # Price momentum
            if len(data) >= 5:
                price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                if abs(price_change) > 0.02:  # 2% change
                    momentum_score += 0.1 if price_change > 0 else -0.1
            
            return min(max(momentum_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {str(e)}")
            return 0.5
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            if 'Volatility' in data.columns:
                current_vol = data['Volatility'].iloc[-1]
                avg_vol = data['Volatility'].rolling(window=20).mean().iloc[-1]
                
                # Higher volatility is better for CFD trading
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
                volatility_score = min(vol_ratio, 2.0) / 2.0  # Normalize to 0-1
                
                return volatility_score
            else:
                # Calculate simple volatility
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    current_vol = returns.std()
                    avg_vol = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else current_vol
                    
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
                    volatility_score = min(vol_ratio, 2.0) / 2.0
                    
                    return volatility_score
                
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.5
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume score"""
        try:
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                
                # Higher volume is better
                if volume_ratio > 1.5:
                    return 1.0
                elif volume_ratio > 1.0:
                    return 0.7
                elif volume_ratio > 0.5:
                    return 0.4
                else:
                    return 0.2
            else:
                # Calculate simple volume ratio
                if len(data) >= 20:
                    current_volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                    
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio > 1.5:
                        return 1.0
                    elif volume_ratio > 1.0:
                        return 0.7
                    elif volume_ratio > 0.5:
                        return 0.4
                    else:
                        return 0.2
                
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating volume score: {str(e)}")
            return 0.5
    
    def _calculate_support_resistance_score(self, data: pd.DataFrame) -> float:
        """Calculate support/resistance score"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Find support and resistance levels
            support_levels = self._find_support_levels(data)
            resistance_levels = self._find_resistance_levels(data)
            
            # Calculate distance to nearest levels
            nearest_support = min([abs(current_price - level) for level in support_levels]) if support_levels else float('inf')
            nearest_resistance = min([abs(current_price - level) for level in resistance_levels]) if resistance_levels else float('inf')
            
            # Score based on proximity to levels
            if nearest_support < float('inf') and nearest_resistance < float('inf'):
                min_distance = min(nearest_support, nearest_resistance)
                price_range = max(resistance_levels) - min(support_levels) if support_levels and resistance_levels else current_price * 0.1
                
                # Closer to levels is better
                proximity_score = 1.0 - (min_distance / price_range)
                return max(proximity_score, 0)
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance score: {str(e)}")
            return 0.5
    
    def _determine_setup_type(self, data: pd.DataFrame, setup_score: float) -> Tuple[str, str, float]:
        """Determine the type and direction of CFD setup"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Get trend direction
            trend_direction = self._get_trend_direction(data)
            
            # Determine setup type based on score and trend
            if setup_score >= 0.8:
                if trend_direction == 'bullish':
                    setup_type = 'Strong Buy'
                    direction = 'long'
                    confidence = setup_score
                elif trend_direction == 'bearish':
                    setup_type = 'Strong Sell'
                    direction = 'short'
                    confidence = setup_score
                else:
                    setup_type = 'Strong Setup'
                    direction = 'neutral'
                    confidence = setup_score
            elif setup_score >= 0.6:
                if trend_direction == 'bullish':
                    setup_type = 'Buy Setup'
                    direction = 'long'
                    confidence = setup_score
                elif trend_direction == 'bearish':
                    setup_type = 'Sell Setup'
                    direction = 'short'
                    confidence = setup_score
                else:
                    setup_type = 'Moderate Setup'
                    direction = 'neutral'
                    confidence = setup_score
            elif setup_score >= 0.4:
                setup_type = 'Weak Setup'
                direction = 'neutral'
                confidence = setup_score
            else:
                setup_type = 'No Setup'
                direction = 'neutral'
                confidence = setup_score
            
            return setup_type, direction, confidence
            
        except Exception as e:
            self.logger.error(f"Error determining setup type: {str(e)}")
            return 'Unknown', 'neutral', 0.0
    
    def _calculate_entry_levels(self, data: pd.DataFrame, direction: str) -> Dict:
        """Calculate entry levels for CFD setup"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if direction == 'long':
                # For long positions, look for pullback entries
                entry_levels = {
                    'aggressive': current_price * 0.995,  # 0.5% below current
                    'moderate': current_price * 0.99,     # 1% below current
                    'conservative': current_price * 0.985  # 1.5% below current
                }
            elif direction == 'short':
                # For short positions, look for bounce entries
                entry_levels = {
                    'aggressive': current_price * 1.005,  # 0.5% above current
                    'moderate': current_price * 1.01,     # 1% above current
                    'conservative': current_price * 1.015  # 1.5% above current
                }
            else:
                # Neutral - use current price
                entry_levels = {
                    'current': current_price
                }
            
            return entry_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating entry levels: {str(e)}")
            return {'current': data['Close'].iloc[-1]}
    
    def _calculate_stop_loss(self, data: pd.DataFrame, direction: str, entry_levels: Dict) -> Dict:
        """Calculate stop loss levels"""
        try:
            if direction == 'long':
                # For long positions, stop loss below entry
                entry = entry_levels.get('moderate', data['Close'].iloc[-1])
                stop_loss = entry * 0.97  # 3% below entry
            elif direction == 'short':
                # For short positions, stop loss above entry
                entry = entry_levels.get('moderate', data['Close'].iloc[-1])
                stop_loss = entry * 1.03  # 3% above entry
            else:
                stop_loss = None
            
            return {'level': stop_loss, 'percentage': 0.03 if stop_loss else None}
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return {'level': None, 'percentage': None}
    
    def _calculate_take_profit(self, data: pd.DataFrame, direction: str, entry_levels: Dict, stop_loss: Dict) -> Dict:
        """Calculate take profit levels"""
        try:
            if direction == 'long' and stop_loss.get('level'):
                entry = entry_levels.get('moderate', data['Close'].iloc[-1])
                stop_level = stop_loss['level']
                
                # Calculate risk
                risk = entry - stop_level
                
                # Take profit at 2:1 risk/reward ratio
                take_profit = entry + (risk * 2)
                
                return {
                    'level': take_profit,
                    'risk_reward_ratio': 2.0,
                    'percentage': ((take_profit - entry) / entry) * 100
                }
                
            elif direction == 'short' and stop_loss.get('level'):
                entry = entry_levels.get('moderate', data['Close'].iloc[-1])
                stop_level = stop_loss['level']
                
                # Calculate risk
                risk = stop_level - entry
                
                # Take profit at 2:1 risk/reward ratio
                take_profit = entry - (risk * 2)
                
                return {
                    'level': take_profit,
                    'risk_reward_ratio': 2.0,
                    'percentage': ((entry - take_profit) / entry) * 100
                }
            
            return {'level': None, 'risk_reward_ratio': None, 'percentage': None}
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return {'level': None, 'risk_reward_ratio': None, 'percentage': None}
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, entry_levels: Dict, stop_loss: Dict, take_profit: Dict) -> Dict:
        """Calculate risk metrics for CFD setup"""
        try:
            current_price = data['Close'].iloc[-1]
            
            if stop_loss.get('level') and take_profit.get('level'):
                entry = entry_levels.get('moderate', current_price)
                stop = stop_loss['level']
                target = take_profit['level']
                
                # Calculate risk metrics
                risk_amount = abs(entry - stop)
                reward_amount = abs(target - entry)
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
                # Calculate position sizing recommendation
                account_risk = 0.02  # 2% risk per trade
                position_size = account_risk / (risk_amount / entry) if risk_amount > 0 else 0
                
                return {
                    'risk_amount': risk_amount,
                    'reward_amount': reward_amount,
                    'risk_reward_ratio': risk_reward_ratio,
                    'position_size_percentage': position_size * 100,
                    'max_loss_percentage': (risk_amount / entry) * 100
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _get_trend_strength(self, data: pd.DataFrame) -> str:
        """Get trend strength description"""
        try:
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                short_ma = data['SMA_20'].iloc[-1]
                long_ma = data['SMA_50'].iloc[-1]
                
                if short_ma > long_ma:
                    return 'Strong Bullish'
                elif short_ma < long_ma:
                    return 'Strong Bearish'
                else:
                    return 'Sideways'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting trend strength: {str(e)}")
            return 'Unknown'
    
    def _get_trend_direction(self, data: pd.DataFrame) -> str:
        """Get trend direction"""
        try:
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                short_ma = data['SMA_20'].iloc[-1]
                long_ma = data['SMA_50'].iloc[-1]
                
                if short_ma > long_ma:
                    return 'bullish'
                elif short_ma < long_ma:
                    return 'bearish'
                else:
                    return 'neutral'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Error getting trend direction: {str(e)}")
            return 'neutral'
    
    def _get_macd_signal(self, data: pd.DataFrame) -> str:
        """Get MACD signal"""
        try:
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                signal = data['MACD_Signal'].iloc[-1]
                
                if macd > signal:
                    return 'Bullish'
                elif macd < signal:
                    return 'Bearish'
                else:
                    return 'Neutral'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting MACD signal: {str(e)}")
            return 'Unknown'
    
    def _get_volatility_trend(self, data: pd.DataFrame) -> str:
        """Get volatility trend"""
        try:
            if 'Volatility' in data.columns and len(data) >= 20:
                current_vol = data['Volatility'].iloc[-1]
                avg_vol = data['Volatility'].rolling(window=20).mean().iloc[-1]
                
                if current_vol > avg_vol * 1.2:
                    return 'Increasing'
                elif current_vol < avg_vol * 0.8:
                    return 'Decreasing'
                else:
                    return 'Stable'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting volatility trend: {str(e)}")
            return 'Unknown'
    
    def _get_volume_trend(self, data: pd.DataFrame) -> str:
        """Get volume trend"""
        try:
            if 'Volume_Ratio' in data.columns:
                volume_ratio = data['Volume_Ratio'].iloc[-1]
                
                if volume_ratio > 1.5:
                    return 'High'
                elif volume_ratio > 1.0:
                    return 'Above Average'
                elif volume_ratio > 0.5:
                    return 'Below Average'
                else:
                    return 'Low'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting volume trend: {str(e)}")
            return 'Unknown'
    
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find support levels"""
        try:
            support_levels = []
            
            # Use recent lows as support levels
            if len(data) >= 20:
                recent_lows = data['Low'].rolling(window=5).min().dropna()
                support_levels = recent_lows.unique().tolist()
                
                # Filter out levels too close to current price
                current_price = data['Close'].iloc[-1]
                support_levels = [level for level in support_levels if level < current_price * 0.95]
            
            return sorted(support_levels)
            
        except Exception as e:
            self.logger.error(f"Error finding support levels: {str(e)}")
            return []
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find resistance levels"""
        try:
            resistance_levels = []
            
            # Use recent highs as resistance levels
            if len(data) >= 20:
                recent_highs = data['High'].rolling(window=5).max().dropna()
                resistance_levels = recent_highs.unique().tolist()
                
                # Filter out levels too close to current price
                current_price = data['Close'].iloc[-1]
                resistance_levels = [level for level in resistance_levels if level > current_price * 1.05]
            
            return sorted(resistance_levels)
            
        except Exception as e:
            self.logger.error(f"Error finding resistance levels: {str(e)}")
            return []
    
    def _get_price_position(self, data: pd.DataFrame) -> str:
        """Get current price position relative to support/resistance"""
        try:
            current_price = data['Close'].iloc[-1]
            support_levels = self._find_support_levels(data)
            resistance_levels = self._find_resistance_levels(data)
            
            if support_levels and resistance_levels:
                nearest_support = max([level for level in support_levels if level < current_price], default=0)
                nearest_resistance = min([level for level in resistance_levels if level > current_price], default=float('inf'))
                
                if nearest_support > 0 and nearest_resistance < float('inf'):
                    range_size = nearest_resistance - nearest_support
                    position = (current_price - nearest_support) / range_size if range_size > 0 else 0.5
                    
                    if position < 0.3:
                        return 'Near Support'
                    elif position > 0.7:
                        return 'Near Resistance'
                    else:
                        return 'Mid Range'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting price position: {str(e)}")
            return 'Unknown'
    
    def get_top_cfd_setups(self, instruments_data: Dict[str, pd.DataFrame], 
                           min_confidence: float = 0.6) -> List[Dict]:
        """
        Get top CFD setups across multiple instruments
        
        Args:
            instruments_data: Dictionary mapping instrument symbols to their data
            min_confidence: Minimum confidence score for setups
        
        Returns:
            List of top CFD setups sorted by confidence
        """
        all_setups = []
        
        for symbol, data in instruments_data.items():
            if not data.empty:
                setup = self.analyze_cfd_setups(data)
                if setup and setup.get('confidence', 0) >= min_confidence:
                    setup['symbol'] = symbol
                    all_setups.append(setup)
        
        # Sort by confidence score (highest first)
        all_setups.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return all_setups
    
    def generate_cfd_report(self, setups: List[Dict]) -> str:
        """Generate a comprehensive CFD trading report"""
        try:
            if not setups:
                return "No CFD setups found matching the criteria."
            
            report = f"""
CFD TRADING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

TOP CFD SETUPS:
"""
            
            for i, setup in enumerate(setups[:10], 1):  # Top 10 setups
                symbol = setup.get('symbol', 'Unknown')
                setup_type = setup.get('setup_type', 'Unknown')
                direction = setup.get('direction', 'neutral')
                confidence = setup.get('confidence', 0)
                setup_score = setup.get('setup_score', 0)
                
                report += f"""
{i}. {symbol} - {setup_type}
   Direction: {direction.upper()}
   Confidence: {confidence:.1%}
   Setup Score: {setup_score:.1%}
"""
                
                # Add entry levels
                entry_levels = setup.get('entry_levels', {})
                if entry_levels:
                    report += "   Entry Levels:\n"
                    for level_type, price in entry_levels.items():
                        report += f"     {level_type.title()}: ${price:.4f}\n"
                
                # Add stop loss and take profit
                stop_loss = setup.get('stop_loss', {})
                take_profit = setup.get('take_profit', {})
                
                if stop_loss.get('level'):
                    report += f"   Stop Loss: ${stop_loss['level']:.4f} ({stop_loss['percentage']:.1%})\n"
                
                if take_profit.get('level'):
                    report += f"   Take Profit: ${take_profit['level']:.4f} ({take_profit['percentage']:.1%})\n"
                
                # Add risk metrics
                risk_metrics = setup.get('risk_metrics', {})
                if risk_metrics:
                    rr_ratio = risk_metrics.get('risk_reward_ratio', 0)
                    position_size = risk_metrics.get('position_size_percentage', 0)
                    report += f"   Risk/Reward: {rr_ratio:.1f}:1\n"
                    report += f"   Position Size: {position_size:.1f}% of account\n"
                
                report += "   " + "-"*60 + "\n"
            
            report += f"""
SUMMARY:
- Total setups found: {len(setups)}
- Top setup confidence: {setups[0].get('confidence', 0):.1%}
- Average confidence: {sum(s.get('confidence', 0) for s in setups) / len(setups):.1%}

RECOMMENDATIONS:
1. Focus on setups with confidence > 70%
2. Use proper position sizing (max 2% risk per trade)
3. Set stop losses and take profits as indicated
4. Monitor setups for changes in market conditions
5. Consider market correlation when trading multiple instruments

DISCLAIMER: This report is for educational purposes only. 
Always conduct your own analysis and risk management.
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating CFD report: {str(e)}")
            return f"Error generating report: {str(e)}"