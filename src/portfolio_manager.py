"""
Portfolio Manager
Handles portfolio construction, optimization, and risk management
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
from src.config import Config

class PortfolioManager:
    """Manages portfolio construction, optimization, and risk management"""
    
    def __init__(self, initial_capital: float = None):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital or self.config.DEFAULT_INITIAL_CAPITAL
        self.portfolio = {}
        self.portfolio_history = []
        self.risk_metrics = {}
        
    def create_portfolio(self, symbols: List[str], weights: List[float] = None,
                        strategy: str = 'equal_weight') -> Dict:
        """
        Create a new portfolio
        
        Args:
            symbols: List of stock symbols
            weights: Initial weights (if None, use strategy)
            strategy: Portfolio strategy ('equal_weight', 'market_cap', 'risk_parity')
        
        Returns:
            Portfolio configuration
        """
        if not symbols:
            self.logger.error("No symbols provided for portfolio")
            return {}
        
        if weights is None:
            weights = self._calculate_weights(symbols, strategy)
        
        # Validate weights
        if len(weights) != len(symbols):
            self.logger.error("Number of weights must match number of symbols")
            return {}
        
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            self.logger.error("Weights must sum to 1.0")
            return {}
        
        # Create portfolio
        self.portfolio = {
            'symbols': symbols,
            'weights': weights,
            'capital': self.initial_capital,
            'positions': {},
            'created_date': pd.Timestamp.now()
        }
        
        # Calculate initial positions
        for symbol, weight in zip(symbols, weights):
            position_value = self.initial_capital * weight
            self.portfolio['positions'][symbol] = {
                'weight': weight,
                'value': position_value,
                'shares': 0,  # Will be calculated when price data is available
                'entry_price': 0,
                'current_price': 0
            }
        
        self.logger.info(f"Portfolio created with {len(symbols)} assets and ${self.initial_capital:,.2f} capital")
        return self.portfolio
    
    def _calculate_weights(self, symbols: List[str], strategy: str) -> List[float]:
        """Calculate portfolio weights based on strategy"""
        n_assets = len(symbols)
        
        if strategy == 'equal_weight':
            return [1.0 / n_assets] * n_assets
        
        elif strategy == 'market_cap':
            # This would require market cap data from API
            # For now, return equal weights
            self.logger.warning("Market cap strategy not implemented, using equal weights")
            return [1.0 / n_assets] * n_assets
        
        elif strategy == 'risk_parity':
            # This would require historical volatility data
            # For now, return equal weights
            self.logger.warning("Risk parity strategy not implemented, using equal weights")
            return [1.0 / n_assets] * n_assets
        
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using equal weights")
            return [1.0 / n_assets] * n_assets
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, 
                          method: str = 'sharpe_ratio',
                          constraints: Dict = None) -> Dict:
        """
        Optimize portfolio weights using various methods
        
        Args:
            returns_data: DataFrame with asset returns
            method: Optimization method ('sharpe_ratio', 'min_variance', 'max_return')
            constraints: Additional constraints for optimization
        
        Returns:
            Optimized portfolio weights
        """
        if returns_data.empty:
            self.logger.error("No returns data provided for optimization")
            return {}
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        # Define objective function based on method
        if method == 'sharpe_ratio':
            objective = lambda w: -self._calculate_sharpe_ratio(w, expected_returns, cov_matrix)
        elif method == 'min_variance':
            objective = lambda w: self._calculate_portfolio_variance(w, cov_matrix)
        elif method == 'max_return':
            objective = lambda w: -self._calculate_portfolio_return(w, expected_returns)
        else:
            self.logger.error(f"Unknown optimization method: {method}")
            return {}
        
        # Set up constraints
        n_assets = len(expected_returns)
        
        # Default constraints
        default_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Add position size constraints
        for i in range(n_assets):
            default_constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i]})  # Non-negative weights
            default_constraints.append({'type': 'ineq', 'fun': lambda w, i=i: self.config.MAX_POSITION_SIZE - w[i]})  # Max position size
        
        # Add custom constraints if provided
        if constraints:
            default_constraints.extend(constraints)
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                constraints=default_constraints,
                bounds=[(0, self.config.MAX_POSITION_SIZE) for _ in range(n_assets)]
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Update portfolio with optimized weights
                self._update_portfolio_weights(optimized_weights)
                
                optimization_result = {
                    'method': method,
                    'weights': optimized_weights.tolist(),
                    'objective_value': result.fun,
                    'success': True,
                    'message': result.message
                }
                
                self.logger.info(f"Portfolio optimized successfully using {method} method")
                return optimization_result
            else:
                self.logger.warning(f"Portfolio optimization failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            self.logger.error(f"Error during portfolio optimization: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def _calculate_portfolio_return(self, weights: np.ndarray, expected_returns: pd.Series) -> float:
        """Calculate expected portfolio return"""
        return np.sum(weights * expected_returns)
    
    def _calculate_portfolio_variance(self, weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
        """Calculate portfolio variance"""
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, expected_returns: pd.Series, 
                                cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        portfolio_return = self._calculate_portfolio_return(weights, expected_returns)
        portfolio_volatility = np.sqrt(self._calculate_portfolio_variance(weights, cov_matrix))
        
        if portfolio_volatility == 0:
            return 0
        
        return (portfolio_return - risk_free_rate) / portfolio_volatility
    
    def _update_portfolio_weights(self, new_weights: np.ndarray):
        """Update portfolio with new weights"""
        if not self.portfolio:
            return
        
        # Update weights
        self.portfolio['weights'] = new_weights.tolist()
        
        # Update position values
        for i, symbol in enumerate(self.portfolio['symbols']):
            if symbol in self.portfolio['positions']:
                self.portfolio['positions'][symbol]['weight'] = new_weights[i]
                self.portfolio['positions'][symbol]['value'] = self.initial_capital * new_weights[i]
    
    def calculate_risk_metrics(self, returns_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive risk metrics for the portfolio
        
        Args:
            returns_data: DataFrame with asset returns
        
        Returns:
            Dictionary with risk metrics
        """
        if returns_data.empty or not self.portfolio:
            return {}
        
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns_data)
            
            # Basic risk metrics
            risk_metrics = {
                'total_return': (portfolio_returns + 1).prod() - 1,
                'annualized_return': (portfolio_returns + 1).prod() ** (252 / len(portfolio_returns)) - 1,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio_simple(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'var_95': self._calculate_value_at_risk(portfolio_returns, 0.95),
                'cvar_95': self._calculate_conditional_var(portfolio_returns, 0.95),
                'skewness': portfolio_returns.skew(),
                'kurtosis': portfolio_returns.kurtosis()
            }
            
            # Store risk metrics
            self.risk_metrics = risk_metrics
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns based on current weights"""
        if not self.portfolio or 'weights' not in self.portfolio:
            return pd.Series()
        
        weights = np.array(self.portfolio['weights'])
        portfolio_returns = (returns_data * weights).sum(axis=1)
        return portfolio_returns
    
    def _calculate_sharpe_ratio_simple(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate simple Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if returns.std() == 0:
            return 0
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_value_at_risk(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def rebalance_portfolio(self, current_prices: Dict[str, float], 
                           rebalance_threshold: float = 0.05) -> Dict:
        """
        Rebalance portfolio if weights deviate significantly
        
        Args:
            current_prices: Current prices for each asset
            rebalance_threshold: Threshold for rebalancing (default: 5%)
        
        Returns:
            Rebalancing actions
        """
        if not self.portfolio:
            return {}
        
        rebalance_actions = {}
        total_value = 0
        
        # Calculate current portfolio value and weights
        for symbol in self.portfolio['symbols']:
            if symbol in current_prices:
                current_price = current_prices[symbol]
                shares = self.portfolio['positions'][symbol]['value'] / current_price
                
                self.portfolio['positions'][symbol]['current_price'] = current_price
                self.portfolio['positions'][symbol]['shares'] = shares
                
                current_value = shares * current_price
                total_value += current_value
        
        # Calculate current weights
        current_weights = {}
        for symbol in self.portfolio['symbols']:
            if symbol in current_prices:
                current_value = self.portfolio['positions'][symbol]['shares'] * current_prices[symbol]
                current_weights[symbol] = current_value / total_value
        
        # Check if rebalancing is needed
        target_weights = dict(zip(self.portfolio['positions'].keys(), self.portfolio['weights']))
        
        for symbol in self.portfolio['symbols']:
            if symbol in current_weights and symbol in target_weights:
                deviation = abs(current_weights[symbol] - target_weights[symbol])
                
                if deviation > rebalance_threshold:
                    # Calculate required trades
                    target_value = total_value * target_weights[symbol]
                    current_value = current_weights[symbol] * total_value
                    trade_value = target_value - current_value
                    
                    rebalance_actions[symbol] = {
                        'current_weight': current_weights[symbol],
                        'target_weight': target_weights[symbol],
                        'deviation': deviation,
                        'trade_value': trade_value,
                        'trade_shares': trade_value / current_prices[symbol]
                    }
        
        if rebalance_actions:
            self.logger.info(f"Portfolio rebalancing required for {len(rebalance_actions)} assets")
        else:
            self.logger.info("Portfolio is within rebalancing threshold")
        
        return rebalance_actions
    
    def add_position(self, symbol: str, weight: float, price: float) -> bool:
        """
        Add a new position to the portfolio
        
        Args:
            symbol: Stock symbol
            weight: Position weight
            price: Entry price
        
        Returns:
            True if successful, False otherwise
        """
        if not self.portfolio:
            self.logger.error("No portfolio exists")
            return False
        
        if symbol in self.portfolio['positions']:
            self.logger.warning(f"Position {symbol} already exists in portfolio")
            return False
        
        # Check if adding this position would exceed max position size
        if weight > self.config.MAX_POSITION_SIZE:
            self.logger.error(f"Position weight {weight:.2%} exceeds maximum {self.config.MAX_POSITION_SIZE:.2%}")
            return False
        
        # Add position
        position_value = self.initial_capital * weight
        shares = position_value / price
        
        self.portfolio['positions'][symbol] = {
            'weight': weight,
            'value': position_value,
            'shares': shares,
            'entry_price': price,
            'current_price': price
        }
        
        self.portfolio['symbols'].append(symbol)
        self.portfolio['weights'].append(weight)
        
        self.logger.info(f"Added position {symbol} with {weight:.2%} weight")
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position from the portfolio
        
        Args:
            symbol: Stock symbol to remove
        
        Returns:
            True if successful, False otherwise
        """
        if not self.portfolio or symbol not in self.portfolio['positions']:
            self.logger.error(f"Position {symbol} not found in portfolio")
            return False
        
        # Remove position
        del self.portfolio['positions'][symbol]
        
        # Remove from symbols and weights
        if symbol in self.portfolio['symbols']:
            idx = self.portfolio['symbols'].index(symbol)
            self.portfolio['symbols'].pop(idx)
            self.portfolio['weights'].pop(idx)
        
        self.logger.info(f"Removed position {symbol} from portfolio")
        return True
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        if not self.portfolio:
            return {}
        
        summary = {
            'total_assets': len(self.portfolio['symbols']),
            'total_capital': self.initial_capital,
            'positions': self.portfolio['positions'],
            'created_date': self.portfolio['created_date'],
            'risk_metrics': self.risk_metrics
        }
        
        return summary
    
    def export_portfolio(self, filename: str, format: str = 'csv') -> bool:
        """
        Export portfolio to file
        
        Args:
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
        
        Returns:
            True if successful, False otherwise
        """
        if not self.portfolio:
            return False
        
        try:
            # Create portfolio DataFrame
            portfolio_data = []
            for symbol, position in self.portfolio['positions'].items():
                portfolio_data.append({
                    'Symbol': symbol,
                    'Weight': position['weight'],
                    'Value': position['value'],
                    'Shares': position['shares'],
                    'Entry_Price': position['entry_price'],
                    'Current_Price': position['current_price']
                })
            
            df = pd.DataFrame(portfolio_data)
            
            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(filename, index=False)
            elif format.lower() == 'excel':
                df.to_excel(filename, index=False, engine='openpyxl')
            elif format.lower() == 'json':
                df.to_json(filename, orient='records')
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Portfolio exported successfully to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting portfolio: {str(e)}")
            return False