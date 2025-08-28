#!/usr/bin/env python3
"""
Financial AI System Demo
Demonstrates the key features and capabilities of the Financial AI platform
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_manager import FinancialDataManager
from src.ai_models import FinancialAIModels
from src.portfolio_manager import PortfolioManager
from src.trading_strategies import TradingStrategies
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def demo_data_management():
    """Demonstrate data management capabilities"""
    print_header("DATA MANAGEMENT DEMO")
    
    try:
        # Initialize data manager
        dm = FinancialDataManager()
        print("✓ Data manager initialized successfully")
        
        # Get stock data for a popular stock
        symbol = "AAPL"
        print_section(f"Fetching data for {symbol}")
        
        data = dm.get_stock_data(symbol, period="6mo")
        if not data.empty:
            print(f"✓ Retrieved {len(data)} data points for {symbol}")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
            print(f"  Technical indicators: {', '.join([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])}")
        else:
            print(f"✗ Failed to retrieve data for {symbol}")
            return False
        
        # Get company information
        print_section("Company Information")
        info = dm.get_company_info(symbol)
        if info:
            print(f"✓ Company: {info.get('name', 'N/A')}")
            print(f"  Sector: {info.get('sector', 'N/A')}")
            print(f"  Market Cap: ${info.get('market_cap', 0) / 1e9:.2f}B")
            print(f"  P/E Ratio: {info.get('pe_ratio', 'N/A')}")
        else:
            print(f"✗ Failed to retrieve company info for {symbol}")
        
        # Get market data
        print_section("Market Overview")
        market_data = dm.get_market_data()
        if market_data:
            print(f"✓ Retrieved data for {len(market_data)} market indices")
            for index, data in market_data.items():
                if not data.empty:
                    latest = data.iloc[-1]
                    print(f"  {index}: ${latest['Close']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data management demo: {e}")
        return False

def demo_ai_models():
    """Demonstrate AI model capabilities"""
    print_header("AI MODELS DEMO")
    
    try:
        # Initialize AI models
        ai = FinancialAIModels()
        print("✓ AI models initialized successfully")
        
        # Get sample data for training
        dm = FinancialDataManager()
        data = dm.get_stock_data("AAPL", period="1y")
        
        if data.empty:
            print("✗ No data available for AI model training")
            return False
        
        print_section("Training Machine Learning Models")
        print("Training Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, and SVR models...")
        
        # Train ML models (with reduced epochs for demo)
        ml_results = ai.train_ml_models(data, test_size=0.2)
        
        if ml_results:
            print(f"✓ Successfully trained {len(ml_results)} ML models")
            for model_name, result in ml_results.items():
                if 'r2' in result:
                    print(f"  {model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
        else:
            print("✗ Failed to train ML models")
        
        print_section("Making Predictions")
        if ml_results:
            # Make predictions with the best performing model
            best_model = max(ml_results.keys(), key=lambda x: ml_results[x].get('r2', 0))
            predictions = ai.predict(best_model, data)
            
            if len(predictions) > 0:
                print(f"✓ Generated {len(predictions)} predictions using {best_model}")
                print(f"  Latest prediction: ${predictions[-1]:.2f}")
                print(f"  Current price: ${data['Close'].iloc[-1]:.2f}")
            else:
                print("✗ Failed to generate predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in AI models demo: {e}")
        return False

def demo_portfolio_management():
    """Demonstrate portfolio management capabilities"""
    print_header("PORTFOLIO MANAGEMENT DEMO")
    
    try:
        # Initialize portfolio manager
        pm = PortfolioManager(initial_capital=100000)
        print("✓ Portfolio manager initialized successfully")
        
        # Create a sample portfolio
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        print_section("Creating Portfolio")
        print(f"Creating portfolio with {len(symbols)} stocks: {', '.join(symbols)}")
        
        portfolio = pm.create_portfolio(symbols, strategy='equal_weight')
        if portfolio:
            print("✓ Portfolio created successfully")
            print(f"  Total assets: {portfolio['total_assets']}")
            print(f"  Total capital: ${portfolio['total_capital']:,.2f}")
            print(f"  Strategy: {portfolio.get('strategy', 'equal_weight')}")
        else:
            print("✗ Failed to create portfolio")
            return False
        
        # Get portfolio summary
        print_section("Portfolio Summary")
        summary = pm.get_portfolio_summary()
        if summary:
            print("✓ Portfolio summary retrieved")
            for symbol, position in summary['positions'].items():
                print(f"  {symbol}: {position['weight']*100:.1f}% (${position['value']:,.2f})")
        
        # Portfolio optimization (simulated)
        print_section("Portfolio Optimization")
        print("Note: Full optimization requires historical returns data")
        print("Portfolio optimization features are available in the web interface")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in portfolio management demo: {e}")
        return False

def demo_trading_strategies():
    """Demonstrate trading strategy capabilities"""
    print_header("TRADING STRATEGIES DEMO")
    
    try:
        # Initialize trading strategies
        ts = TradingStrategies()
        print("✓ Trading strategies initialized successfully")
        
        # Get sample data
        dm = FinancialDataManager()
        data = dm.get_stock_data("AAPL", period="6mo")
        
        if data.empty:
            print("✗ No data available for trading strategies")
            return False
        
        print_section("Generating Trading Signals")
        
        # Test different strategies
        strategies = ['moving_average_crossover', 'rsi_strategy', 'bollinger_bands', 'combined']
        
        for strategy in strategies:
            print(f"Testing {strategy} strategy...")
            signals = ts.generate_signals(data, strategy=strategy)
            
            if not signals.empty:
                # Get latest signal
                signal_columns = [col for col in signals.columns if 'Signal' in col and col != 'Signal']
                if signal_columns:
                    latest_signal = signals.iloc[-1]
                    for col in signal_columns:
                        if col in latest_signal:
                            signal_value = latest_signal[col]
                            if signal_value == 1:
                                signal_text = "BUY"
                            elif signal_value == -1:
                                signal_text = "SELL"
                            else:
                                signal_text = "HOLD"
                            print(f"  {col}: {signal_text}")
                print(f"✓ {strategy} strategy completed")
            else:
                print(f"✗ Failed to generate signals for {strategy}")
        
        print_section("Trading Recommendations")
        recommendations = ts.get_trading_recommendations(data, strategy='combined')
        if recommendations:
            print("✓ Trading recommendations generated")
            print(f"  Action: {recommendations['action']}")
            print(f"  Confidence: {recommendations['confidence']}")
            print(f"  Current Price: ${recommendations['current_price']:.2f}")
            print(f"  Price Change: {recommendations['price_change']*100:.2f}%")
        else:
            print("✗ Failed to generate trading recommendations")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in trading strategies demo: {e}")
        return False

def demo_web_interface():
    """Demonstrate web interface capabilities"""
    print_header("WEB INTERFACE DEMO")
    
    print("The Financial AI system includes a comprehensive web interface with:")
    print("✓ Real-time market dashboard")
    print("✓ Interactive stock charts")
    print("✓ Portfolio management tools")
    print("✓ Trading signal generation")
    print("✓ AI model training interface")
    print("✓ Strategy backtesting")
    print("✓ Risk analysis and reporting")
    
    print("\nTo access the web interface:")
    print("1. Run: python app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Navigate through the different sections")
    
    print("\nKey web features:")
    print("• Responsive design for all devices")
    print("• Real-time data updates")
    print("• Interactive Chart.js visualizations")
    print("• RESTful API endpoints")
    print("• Modern Bootstrap 5 UI")
    
    return True

def main():
    """Run the complete demo"""
    print_header("FINANCIAL AI SYSTEM DEMO")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This demo showcases the key capabilities of the Financial AI platform")
    
    # Run all demos
    demos = [
        ("Data Management", demo_data_management),
        ("AI Models", demo_ai_models),
        ("Portfolio Management", demo_portfolio_management),
        ("Trading Strategies", demo_trading_strategies),
        ("Web Interface", demo_web_interface)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} Running {demo_name} Demo {'='*20}")
            results[demo_name] = demo_func()
        except Exception as e:
            logger.error(f"Demo {demo_name} failed: {e}")
            results[demo_name] = False
    
    # Summary
    print_header("DEMO SUMMARY")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"Completed {total_demos} demos: {successful_demos} successful, {total_demos - successful_demos} failed")
    
    for demo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {demo_name}: {status}")
    
    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("The Financial AI system is ready to use.")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed.")
        print("Check the logs above for error details.")
    
    print("\nNext steps:")
    print("1. Review the generated data and models")
    print("2. Explore the web interface")
    print("3. Customize configuration in .env file")
    print("4. Train models on your preferred stocks")
    print("5. Build and optimize portfolios")
    
    print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        sys.exit(1)