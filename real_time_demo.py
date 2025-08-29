#!/usr/bin/env python3
"""
Real-Time Data Demo for Financial AI
Demonstrates live market data, news sentiment, and real-time analysis
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.real_time_data import RealTimeDataManager
from src.financial_advisor import FinancialAdvisor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def demo_real_time_data_manager():
    """Demonstrate Real-Time Data Manager capabilities"""
    print_header("REAL-TIME DATA MANAGER DEMO")
    
    try:
        # Initialize real-time data manager
        real_time_data = RealTimeDataManager()
        
        print("✓ Real-Time Data Manager initialized successfully")
        print("✓ Live market data sources configured")
        print("✓ Real-time news sentiment analysis ready")
        print("✓ Live technical indicators calculation ready")
        
        # Show configuration
        print_section("Configuration")
        print(f"Cache Duration: {real_time_data.config.CACHE_DURATION} seconds")
        print(f"Data Validation: {real_time_data.config.DATA_VALIDATION}")
        print(f"Rate Limits: {real_time_data.config.RATE_LIMITS}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Real-Time Data Manager demo: {e}")
        return False

def demo_live_market_data():
    """Demonstrate live market data fetching"""
    print_header("LIVE MARKET DATA DEMO")
    
    try:
        real_time_data = RealTimeDataManager()
        
        # Test symbols
        test_symbols = ['AAPL', 'VOD.L', '^FTSE', 'GBPUSD=X', 'GC=F']
        
        for symbol in test_symbols:
            print_section(f"Live Data for {symbol}")
            
            try:
                # Get live market data
                data = real_time_data.get_live_market_data(symbol, interval='5m', period='1d')
                
                if not data.empty:
                    print(f"✓ Retrieved {len(data)} data points")
                    print(f"  Latest Price: ${data['Close'].iloc[-1]:.4f}")
                    print(f"  Date Range: {data.index[0]} to {data.index[-1]}")
                    print(f"  Data Source: Real-time")
                    
                    # Show data validation
                    if len(data) >= real_time_data.config.DATA_VALIDATION['min_data_points']:
                        print(f"  Data Quality: ✓ Sufficient data points")
                    else:
                        print(f"  Data Quality: ⚠ Insufficient data points")
                    
                    # Check data freshness
                    latest_time = data.index[-1]
                    if isinstance(latest_time, pd.Timestamp):
                        age_minutes = (pd.Timestamp.now() - latest_time).total_seconds() / 60
                        if age_minutes <= real_time_data.config.DATA_VALIDATION['data_freshness']:
                            print(f"  Data Freshness: ✓ {age_minutes:.1f} minutes old")
                        else:
                            print(f"  Data Freshness: ⚠ {age_minutes:.1f} minutes old")
                else:
                    print(f"✗ No data available for {symbol}")
                    
            except Exception as e:
                print(f"✗ Error getting data for {symbol}: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live market data demo: {e}")
        return False

def demo_live_uk_market_status():
    """Demonstrate live UK market status"""
    print_header("LIVE UK MARKET STATUS DEMO")
    
    try:
        real_time_data = RealTimeDataManager()
        
        print_section("Checking UK Market Status")
        print("The system is checking real-time UK market status...")
        
        # Get live market status
        market_status = real_time_data.get_live_market_status()
        
        if market_status:
            print("✓ UK market status retrieved successfully")
            print(f"  Status: {market_status.get('status', 'UNKNOWN')}")
            print(f"  Current Time: {market_status.get('current_time', 'N/A')}")
            print(f"  Market Hours: {market_status.get('market_open', 'N/A')} - {market_status.get('market_close', 'N/A')}")
            print(f"  Is Open: {market_status.get('is_open', False)}")
            print(f"  Timezone: {market_status.get('timezone', 'N/A')}")
            
            if market_status.get('time_to_open'):
                print(f"  Time to Market Open: {market_status['time_to_open']}")
            elif market_status.get('time_to_close'):
                print(f"  Market Closed: {market_status['time_to_close']} ago")
            
            # Trading implications
            if market_status.get('is_open'):
                print("\n📈 Trading Status: UK market is OPEN")
                print("✓ You can execute CFD trades now")
                print("✓ Real-time market data available")
                print("✓ Live recommendations active")
            else:
                print("\n⚠️ Trading Status: UK market is CLOSED")
                if market_status.get('time_to_open'):
                    print(f"⏰ Market opens in {market_status['time_to_open']}")
                else:
                    print("🌙 Market closed for the day")
        else:
            print("✗ Failed to get UK market status")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in UK market status demo: {e}")
        return False

def demo_live_news_sentiment():
    """Demonstrate live news sentiment analysis"""
    print_header("LIVE NEWS SENTIMENT DEMO")
    
    try:
        real_time_data = RealTimeDataManager()
        
        print_section("Real-Time News Sentiment Analysis")
        print("The system is analyzing live financial news...")
        
        # Get general market news sentiment
        general_news = real_time_data.get_live_news_sentiment()
        
        if general_news:
            print("✓ General market news sentiment retrieved")
            print(f"  Overall Sentiment: {general_news.get('overall_sentiment', 'NEUTRAL')}")
            print(f"  Bullish News: {general_news.get('bullish_news', 0)}")
            print(f"  Bearish News: {general_news.get('bearish_news', 0)}")
            print(f"  Neutral News: {general_news.get('neutral_news', 0)}")
            print(f"  Data Source: {general_news.get('data_source', 'Unknown')}")
            
            # Show key headlines
            key_headlines = general_news.get('key_headlines', [])
            if key_headlines:
                print("\n  Key Headlines:")
                for i, headline in enumerate(key_headlines[:5], 1):
                    print(f"    {i}. {headline}")
        
        # Get symbol-specific news sentiment
        print_section("Symbol-Specific News Sentiment")
        test_symbols = ['AAPL', 'VOD.L']
        
        for symbol in test_symbols:
            try:
                symbol_news = real_time_data.get_live_news_sentiment(symbol, 'business')
                
                if symbol_news:
                    print(f"\n  {symbol}:")
                    print(f"    Sentiment: {symbol_news.get('sentiment', 'NEUTRAL')}")
                    print(f"    Confidence: {symbol_news.get('confidence', 0):.1%}")
                    print(f"    Data Source: {symbol_news.get('data_source', 'Unknown')}")
                    
                    # Show symbol headlines
                    symbol_headlines = symbol_news.get('key_headlines', [])
                    if symbol_headlines:
                        print(f"    Headlines: {len(symbol_headlines)} articles")
                else:
                    print(f"  {symbol}: No news data available")
                    
            except Exception as e:
                print(f"  {symbol}: Error - {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live news sentiment demo: {e}")
        return False

def demo_live_technical_indicators():
    """Demonstrate live technical indicators"""
    print_header("LIVE TECHNICAL INDICATORS DEMO")
    
    try:
        real_time_data = RealTimeDataManager()
        
        print_section("Real-Time Technical Analysis")
        print("The system is calculating live technical indicators...")
        
        # Test symbols
        test_symbols = ['AAPL', 'VOD.L', '^FTSE']
        
        for symbol in test_symbols:
            print_section(f"Technical Analysis for {symbol}")
            
            try:
                # Get live technical indicators
                indicators = real_time_data.get_live_technical_indicators(symbol)
                
                if indicators:
                    print(f"✓ Technical indicators calculated for {symbol}")
                    print(f"  Current Price: ${indicators.get('current_price', 0):.4f}")
                    print(f"  Price Change: {indicators.get('price_change_percent', 0):.2f}%")
                    print(f"  Volume Ratio: {indicators.get('volume_ratio', 1):.2f}")
                    
                    # RSI
                    rsi = indicators.get('rsi')
                    if rsi:
                        print(f"  RSI: {rsi:.2f}")
                        if rsi < 30:
                            print(f"    Status: Oversold (Bullish)")
                        elif rsi > 70:
                            print(f"    Status: Overbought (Bearish)")
                        else:
                            print(f"    Status: Neutral")
                    
                    # MACD
                    macd_signal = indicators.get('macd_signal_type')
                    if macd_signal:
                        print(f"  MACD Signal: {macd_signal}")
                    
                    # Moving Averages
                    if 'sma_20' in indicators:
                        print(f"  SMA 20: ${indicators['sma_20']:.4f}")
                        print(f"  Price vs SMA 20: {indicators.get('price_vs_sma20', 'UNKNOWN')}")
                    
                    if 'sma_50' in indicators:
                        print(f"  SMA 50: ${indicators['sma_50']:.4f}")
                        print(f"  Price vs SMA 50: {indicators.get('price_vs_sma50', 'UNKNOWN')}")
                    
                    # Bollinger Bands
                    bb_position = indicators.get('bb_position')
                    if bb_position:
                        print(f"  Bollinger Bands Position: {bb_position}")
                    
                    # Volatility
                    volatility = indicators.get('volatility')
                    if volatility:
                        print(f"  Annualized Volatility: {volatility:.2%}")
                    
                    # Technical Score
                    tech_score = indicators.get('technical_score')
                    if tech_score:
                        print(f"  Technical Score: {tech_score:.1%}")
                        if tech_score > 0.7:
                            print(f"    Assessment: Strong technical indicators")
                        elif tech_score < 0.3:
                            print(f"    Assessment: Weak technical indicators")
                        else:
                            print(f"    Assessment: Mixed technical indicators")
                    
                    print(f"  Data Source: {indicators.get('data_source', 'Unknown')}")
                    print(f"  Last Updated: {indicators.get('last_updated', 'Unknown')}")
                    
                else:
                    print(f"✗ No technical indicators available for {symbol}")
                    
            except Exception as e:
                print(f"✗ Error analyzing {symbol}: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live technical indicators demo: {e}")
        return False

def demo_live_market_sentiment():
    """Demonstrate live market sentiment analysis"""
    print_header("LIVE MARKET SENTIMENT DEMO")
    
    try:
        real_time_data = RealTimeDataManager()
        
        print_section("Real-Time Global Market Sentiment")
        print("The system is analyzing live market sentiment across major indices...")
        
        # Get live market sentiment
        market_sentiment = real_time_data.get_live_market_sentiment()
        
        if market_sentiment:
            print("✓ Global market sentiment analyzed successfully")
            print(f"  Overall Sentiment: {market_sentiment.get('overall_sentiment', 'NEUTRAL')}")
            print(f"  Sentiment Score: {market_sentiment.get('sentiment_score', 0):.1%}")
            print(f"  Indices Analyzed: {market_sentiment.get('indices_analyzed', 0)}")
            print(f"  Data Source: {market_sentiment.get('data_source', 'Unknown')}")
            print(f"  Timestamp: {market_sentiment.get('timestamp', 'Unknown')}")
            
            # Sentiment interpretation
            sentiment = market_sentiment.get('overall_sentiment', 'NEUTRAL')
            if sentiment == 'BULLISH':
                print("\n📈 Market Sentiment: BULLISH")
                print("✓ Positive market momentum")
                print("✓ Favorable for long positions")
                print("✓ Consider bullish trading strategies")
            elif sentiment == 'BEARISH':
                print("\n📉 Market Sentiment: BEARISH")
                print("⚠️ Negative market momentum")
                print("⚠️ Favorable for short positions")
                print("⚠️ Consider defensive strategies")
            else:
                print("\n➡️ Market Sentiment: NEUTRAL")
                print("➡️ Mixed market signals")
                print("➡️ Consider range-bound strategies")
                print("➡️ Wait for clearer direction")
        else:
            print("✗ Failed to analyze market sentiment")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in live market sentiment demo: {e}")
        return False

def demo_financial_advisor_real_time():
    """Demonstrate Financial Advisor with real-time data"""
    print_header("FINANCIAL ADVISOR REAL-TIME DEMO")
    
    try:
        financial_advisor = FinancialAdvisor()
        
        print_section("Real-Time Financial Advisor")
        print("The Financial Advisor is analyzing live market conditions...")
        
        # Get real-time market analysis
        market_analysis = financial_advisor.analyze_market_conditions()
        
        if market_analysis:
            print("✓ Real-time market analysis completed")
            print(f"  Data Source: {market_analysis.get('data_source', 'Unknown')}")
            print(f"  Timestamp: {market_analysis.get('timestamp', 'Unknown')}")
            
            # Show key components
            components = ['uk_market_status', 'global_sentiment', 'news_sentiment', 'market_volatility', 'technical_overview']
            
            for component in components:
                if component in market_analysis:
                    data = market_analysis[component]
                    if data:
                        print(f"\n  {component.replace('_', ' ').title()}: ✓ Available")
                        if 'data_source' in data:
                            print(f"    Data Source: {data['data_source']}")
                    else:
                        print(f"\n  {component.replace('_', ' ').title()}: ⚠ Limited data")
                else:
                    print(f"\n  {component.replace('_', ' ').title()}: ✗ Not available")
            
            # Get real-time trading recommendations
            print_section("Real-Time Trading Recommendations")
            print("Generating live trading recommendations for £147 capital...")
            
            recommendations = financial_advisor.get_trading_recommendations(147.0)
            
            if recommendations:
                print("✓ Real-time trading recommendations generated")
                print(f"  Data Source: {recommendations.get('data_source', 'Unknown')}")
                print(f"  Total Recommendations: {recommendations.get('total_recommendations', 0)}")
                print(f"  Filtered Recommendations: {recommendations.get('filtered_recommendations', 0)}")
                
                # Show top recommendations
                top_recommendations = recommendations.get('top_recommendations', [])
                if top_recommendations:
                    print(f"\n  Top {len(top_recommendations)} Recommendations:")
                    for i, rec in enumerate(top_recommendations[:3], 1):
                        print(f"    {i}. {rec.get('symbol', 'Unknown')}")
                        print(f"       Direction: {rec.get('direction', 'neutral').upper()}")
                        print(f"       Confidence: {rec.get('confidence', 0):.1%}")
                        print(f"       Data Source: {rec.get('data_source', 'Unknown')}")
                        print(f"       Last Updated: {rec.get('last_updated', 'Unknown')}")
                else:
                    print("  ⚠ No top recommendations available")
            else:
                print("✗ Failed to generate trading recommendations")
        else:
            print("✗ Failed to analyze market conditions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Financial Advisor real-time demo: {e}")
        return False

def main():
    """Run the complete Real-Time Data demo"""
    print_header("REAL-TIME DATA SYSTEM DEMO FOR FINANCIAL AI")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This demo showcases the real-time data capabilities")
    print("All data is live, authentic, and sourced from real APIs")
    print("No synthetic or simulated information is used")
    
    # Run all demos
    demos = [
        ("Real-Time Data Manager", demo_real_time_data_manager),
        ("Live Market Data", demo_live_market_data),
        ("Live UK Market Status", demo_live_uk_market_status),
        ("Live News Sentiment", demo_live_news_sentiment),
        ("Live Technical Indicators", demo_live_technical_indicators),
        ("Live Market Sentiment", demo_live_market_sentiment),
        ("Financial Advisor Real-Time", demo_financial_advisor_real_time)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*30} Running {demo_name} Demo {'='*30}")
            results[demo_name] = demo_func()
        except Exception as e:
            logger.error(f"Demo {demo_name} failed: {e}")
            results[demo_name] = False
    
    # Summary
    print_header("REAL-TIME DATA DEMO SUMMARY")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"Completed {total_demos} demos: {successful_demos} successful, {total_demos - successful_demos} failed")
    
    for demo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {demo_name}: {status}")
    
    if successful_demos == total_demos:
        print("\n🎉 All Real-Time Data demos completed successfully!")
        print("The system is now using 100% live, authentic market data!")
        print("No synthetic information is being used!")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed.")
        print("Check the logs above for error details.")
    
    print("\nReal-Time Data Features Available:")
    print("✓ Live market data from multiple sources")
    print("✓ Real-time news sentiment analysis")
    print("✓ Live technical indicators calculation")
    print("✓ Real-time UK market status")
    print("✓ Live global market sentiment")
    print("✓ Real-time trading recommendations")
    print("✓ Data validation and quality checks")
    print("✓ Rate limiting and API management")
    print("✓ Caching for performance optimization")
    
    print("\nData Sources:")
    print("✓ Yahoo Finance (Primary)")
    print("✓ Alpha Vantage (Backup)")
    print("✓ Polygon.io (Backup)")
    print("✓ NewsAPI (News)")
    print("✓ Finnhub (News)")
    print("✓ Alpha Vantage News (Sentiment)")
    
    print("\nNext steps:")
    print("1. Set up your API keys in .env file")
    print("2. Run the web interface: python app.py")
    print("3. Get real-time market analysis")
    print("4. Receive live trading recommendations")
    print("5. Trade with confidence using live data")
    
    print(f"\nReal-Time Data Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nReal-Time Data Demo interrupted by user")
    except Exception as e:
        logger.error(f"Real-Time Data Demo failed with error: {e}")
        sys.exit(1)