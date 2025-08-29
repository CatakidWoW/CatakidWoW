"""
Trading Strategies
Implements various trading strategies using technical analysis and AI signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from src.config import Config

class TradingStrategies:
    """Implements various trading strategies and signal generation"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, data: pd.DataFrame, strategy: str = 'combined',
                        params: Dict = None) -> pd.DataFrame:
        """
        Generate trading signals using specified strategy
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            strategy: Trading strategy to use
            params: Strategy parameters
        
        Returns:
            DataFrame with trading signals
        """
        if data.empty:
            return pd.DataFrame()
        
        # Create copy of data to avoid modifying original
        signals = data.copy()
        
        if strategy == 'moving_average_crossover':
            signals = self._moving_average_crossover(signals, params)
        elif strategy == 'rsi_strategy':
            signals = self._rsi_strategy(signals, params)
        elif strategy == 'bollinger_bands':
            signals = self._bollinger_bands_strategy(signals, params)
        elif strategy == 'macd_strategy':
            signals = self._macd_strategy(signals, params)
        elif strategy == 'volume_price_trend':
            signals = self._volume_price_trend_strategy(signals, params)
        elif strategy == 'mean_reversion':
            signals = self._mean_reversion_strategy(signals, params)
        elif strategy == 'momentum_strategy':
            signals = self._momentum_strategy(signals, params)
        elif strategy == 'combined':
            signals = self._combined_strategy(signals, params)
        else:
            self.logger.error(f"Unknown strategy: {strategy}")
            return pd.DataFrame()
        
        return signals
    
    def _moving_average_crossover(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Moving average crossover strategy"""
        if params is None:
            params = {}
        
        short_ma = params.get('short_ma', self.config.SHORT_TERM_MA)
        long_ma = params.get('long_ma', self.config.LONG_TERM_MA)
        
        # Calculate moving averages if not present
        if f'SMA_{short_ma}' not in data.columns:
            data[f'SMA_{short_ma}'] = data['Close'].rolling(window=short_ma).mean()
        if f'SMA_{long_ma}' not in data.columns:
            data[f'SMA_{long_ma}'] = data['Close'].rolling(window=long_ma).mean()
        
        # Generate signals
        data['MA_Signal'] = 0
        data.loc[data[f'SMA_{short_ma}'] > data[f'SMA_{long_ma}'], 'MA_Signal'] = 1  # Buy
        data.loc[data[f'SMA_{short_ma}'] < data[f'SMA_{long_ma}'], 'MA_Signal'] = -1  # Sell
        
        # Signal changes
        data['MA_Signal_Change'] = data['MA_Signal'].diff()
        
        return data
    
    def _rsi_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """RSI-based trading strategy"""
        if params is None:
            params = {}
        
        rsi_period = params.get('rsi_period', self.config.RSI_PERIOD)
        oversold = params.get('oversold', self.config.RSI_OVERSOLD)
        overbought = params.get('overbought', self.config.RSI_OVERBOUGHT)
        
        # Calculate RSI if not present
        if 'RSI' not in data.columns:
            data['RSI'] = self._calculate_rsi(data['Close'], rsi_period)
        
        # Generate signals
        data['RSI_Signal'] = 0
        
        # Oversold condition (buy signal)
        data.loc[data['RSI'] < oversold, 'RSI_Signal'] = 1
        
        # Overbought condition (sell signal)
        data.loc[data['RSI'] > overbought, 'RSI_Signal'] = -1
        
        # Signal changes
        data['RSI_Signal_Change'] = data['RSI_Signal'].diff()
        
        return data
    
    def _bollinger_bands_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Bollinger Bands trading strategy"""
        if params is None:
            params = {}
        
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2)
        
        # Calculate Bollinger Bands if not present
        if 'BB_Upper' not in data.columns:
            data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(
                data['Close'], bb_period, bb_std
            )
        
        # Generate signals
        data['BB_Signal'] = 0
        
        # Price below lower band (buy signal)
        data.loc[data['Close'] < data['BB_Lower'], 'BB_Signal'] = 1
        
        # Price above upper band (sell signal)
        data.loc[data['Close'] > data['BB_Upper'], 'BB_Signal'] = -1
        
        # Signal changes
        data['BB_Signal_Change'] = data['BB_Signal'].diff()
        
        return data
    
    def _macd_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """MACD trading strategy"""
        if params is None:
            params = {}
        
        # Calculate MACD if not present
        if 'MACD' not in data.columns:
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Generate signals
        data['MACD_Signal_Output'] = 0
        
        # MACD crosses above signal line (buy)
        data.loc[(data['MACD'] > data['MACD_Signal']) & 
                 (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)), 'MACD_Signal_Output'] = 1
        
        # MACD crosses below signal line (sell)
        data.loc[(data['MACD'] < data['MACD_Signal']) & 
                 (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)), 'MACD_Signal_Output'] = -1
        
        # Signal changes
        data['MACD_Signal_Change'] = data['MACD_Signal_Output'].diff()
        
        return data
    
    def _volume_price_trend_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Volume-Price Trend strategy"""
        if params is None:
            params = {}
        
        volume_threshold = params.get('volume_threshold', 1.5)
        
        # Calculate volume indicators if not present
        if 'Volume_SMA' not in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        if 'Volume_Ratio' not in data.columns:
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Generate signals
        data['VPT_Signal'] = 0
        
        # High volume with price increase (buy signal)
        data.loc[(data['Volume_Ratio'] > volume_threshold) & 
                 (data['Close'] > data['Close'].shift(1)), 'VPT_Signal'] = 1
        
        # High volume with price decrease (sell signal)
        data.loc[(data['Volume_Ratio'] > volume_threshold) & 
                 (data['Close'] < data['Close'].shift(1)), 'VPT_Signal'] = -1
        
        # Signal changes
        data['VPT_Signal_Change'] = data['VPT_Signal'].diff()
        
        return data
    
    def _mean_reversion_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Mean reversion strategy"""
        if params is None:
            params = {}
        
        lookback_period = params.get('lookback_period', 20)
        std_dev_threshold = params.get('std_dev_threshold', 2)
        
        # Calculate mean reversion indicators
        data['Price_MA'] = data['Close'].rolling(window=lookback_period).mean()
        data['Price_Std'] = data['Close'].rolling(window=lookback_period).std()
        data['Price_ZScore'] = (data['Close'] - data['Price_MA']) / data['Price_Std']
        
        # Generate signals
        data['MR_Signal'] = 0
        
        # Price significantly below mean (buy signal)
        data.loc[data['Price_ZScore'] < -std_dev_threshold, 'MR_Signal'] = 1
        
        # Price significantly above mean (sell signal)
        data.loc[data['Price_ZScore'] > std_dev_threshold, 'MR_Signal'] = -1
        
        # Signal changes
        data['MR_Signal_Change'] = data['MR_Signal'].diff()
        
        return data
    
    def _momentum_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Momentum-based trading strategy"""
        if params is None:
            params = {}
        
        momentum_period = params.get('momentum_period', 10)
        momentum_threshold = params.get('momentum_threshold', 0.02)
        
        # Calculate momentum indicators
        data['Returns'] = data['Close'].pct_change()
        data['Momentum'] = data['Returns'].rolling(window=momentum_period).sum()
        
        # Generate signals
        data['Momentum_Signal'] = 0
        
        # Strong positive momentum (buy signal)
        data.loc[data['Momentum'] > momentum_threshold, 'Momentum_Signal'] = 1
        
        # Strong negative momentum (sell signal)
        data.loc[data['Momentum'] < -momentum_threshold, 'Momentum_Signal'] = -1
        
        # Signal changes
        data['Momentum_Signal_Change'] = data['Momentum_Signal'].diff()
        
        return data
    
    def _combined_strategy(self, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """Combined strategy using multiple signals"""
        if params is None:
            params = {}
        
        # Apply individual strategies
        data = self._moving_average_crossover(data, params)
        data = self._rsi_strategy(data, params)
        data = self._bollinger_bands_strategy(data, params)
        data = self._macd_strategy(data, params)
        data = self._volume_price_trend_strategy(data, params)
        data = self._mean_reversion_strategy(data, params)
        data = self._momentum_strategy(data, params)
        
        # Combine signals with weights
        signal_columns = ['MA_Signal', 'RSI_Signal', 'BB_Signal', 'MACD_Signal_Output', 
                         'VPT_Signal', 'MR_Signal', 'Momentum_Signal']
        
        # Default weights (can be customized)
        weights = params.get('signal_weights', [0.2, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1])
        
        # Ensure weights sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Calculate combined signal
        data['Combined_Signal'] = 0
        for i, col in enumerate(signal_columns):
            if col in data.columns:
                data['Combined_Signal'] += weights[i] * data[col]
        
        # Generate final signal based on threshold
        threshold = params.get('signal_threshold', 0.3)
        data['Final_Signal'] = 0
        data.loc[data['Combined_Signal'] > threshold, 'Final_Signal'] = 1
        data.loc[data['Combined_Signal'] < -threshold, 'Final_Signal'] = -1
        
        # Signal changes
        data['Final_Signal_Change'] = data['Final_Signal'].diff()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                   std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def backtest_strategy(self, data: pd.DataFrame, strategy: str = 'combined',
                         initial_capital: float = 100000, commission: float = 0.001,
                         params: Dict = None) -> Dict:
        """
        Backtest a trading strategy
        
        Args:
            data: DataFrame with OHLCV data
            strategy: Trading strategy to backtest
            initial_capital: Initial capital for backtesting
            commission: Commission rate per trade
            params: Strategy parameters
        
        Returns:
            Backtest results
        """
        if data.empty:
            return {}
        
        # Generate signals
        signals = self.generate_signals(data, strategy, params)
        
        if signals.empty:
            return {}
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # Get signal column
        signal_col = 'Final_Signal' if strategy == 'combined' else f'{strategy.split("_")[0].upper()}_Signal'
        
        if signal_col not in signals.columns:
            self.logger.error(f"Signal column {signal_col} not found")
            return {}
        
        # Run backtest
        for i in range(1, len(signals)):
            current_signal = signals.iloc[i][signal_col]
            current_price = signals.iloc[i]['Close']
            current_date = signals.index[i]
            
            # Check for signal changes
            if current_signal != signals.iloc[i-1][signal_col]:
                if current_signal == 1 and position <= 0:  # Buy signal
                    if position < 0:  # Close short position
                        close_value = abs(position) * current_price * (1 - commission)
                        capital += close_value
                        trades.append({
                            'date': current_date,
                            'action': 'close_short',
                            'price': current_price,
                            'shares': abs(position),
                            'value': close_value,
                            'capital': capital
                        })
                        position = 0
                    
                    # Open long position
                    shares = capital / current_price
                    position = shares
                    capital = 0
                    trades.append({
                        'date': current_date,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price,
                        'capital': capital
                    })
                
                elif current_signal == -1 and position >= 0:  # Sell signal
                    if position > 0:  # Close long position
                        close_value = position * current_price * (1 - commission)
                        capital = close_value
                        trades.append({
                            'date': current_date,
                            'action': 'sell',
                            'price': current_price,
                            'shares': position,
                            'value': close_value,
                            'capital': capital
                        })
                        position = 0
                    
                    # Open short position
                    shares = capital / current_price
                    position = -shares
                    capital = 0
                    trades.append({
                        'date': current_date,
                        'action': 'short',
                        'price': current_price,
                        'shares': shares,
                        'value': shares * current_price,
                        'capital': capital
                    })
            
            # Calculate current equity
            if position > 0:  # Long position
                equity = position * current_price
            elif position < 0:  # Short position
                equity = capital + abs(position) * (2 * signals.iloc[i-1]['Close'] - current_price)
            else:  # No position
                equity = capital
            
            equity_curve.append({
                'date': current_date,
                'equity': equity,
                'position': position,
                'price': current_price
            })
        
        # Calculate performance metrics
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # Calculate returns
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Performance metrics
            total_return = (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital
            annualized_return = (1 + total_return) ** (252 / len(equity_df)) - 1
            volatility = equity_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            
            # Trade statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['action'] in ['sell', 'close_short']])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            results = {
                'strategy': strategy,
                'initial_capital': initial_capital,
                'final_equity': equity_df['equity'].iloc[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'trades': trades,
                'equity_curve': equity_df
            }
            
            return results
        
        return {}
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = equity
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_trading_recommendations(self, data: pd.DataFrame, strategy: str = 'combined',
                                   params: Dict = None) -> Dict:
        """
        Get current trading recommendations
        
        Args:
            data: DataFrame with OHLCV data
            strategy: Trading strategy to use
            params: Strategy parameters
        
        Returns:
            Trading recommendations
        """
        if data.empty:
            return {}
        
        # Generate signals
        signals = self.generate_signals(data, strategy, params)
        
        if signals.empty:
            return {}
        
        # Get latest signals
        latest = signals.iloc[-1]
        
        # Determine recommendation
        if strategy == 'combined':
            signal = latest.get('Final_Signal', 0)
            signal_strength = latest.get('Combined_Signal', 0)
        else:
            signal_col = f'{strategy.split("_")[0].upper()}_Signal'
            signal = latest.get(signal_col, 0)
            signal_strength = 0
        
        # Generate recommendation
        if signal == 1:
            action = 'BUY'
            confidence = 'Strong' if abs(signal_strength) > 0.5 else 'Moderate'
        elif signal == -1:
            action = 'SELL'
            confidence = 'Strong' if abs(signal_strength) > 0.5 else 'Moderate'
        else:
            action = 'HOLD'
            confidence = 'Neutral'
        
        # Additional analysis
        current_price = latest['Close']
        price_change = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        
        recommendation = {
            'action': action,
            'confidence': confidence,
            'current_price': current_price,
            'price_change': price_change,
            'signal_strength': signal_strength,
            'timestamp': latest.name,
            'strategy': strategy,
            'technical_indicators': {
                'rsi': latest.get('RSI', 'N/A'),
                'macd': latest.get('MACD', 'N/A'),
                'volume_ratio': latest.get('Volume_Ratio', 'N/A')
            }
        }
        
        return recommendation