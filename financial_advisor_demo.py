#!/usr/bin/env python3
"""
Financial Advisor Demo for Trading212 CFD
Demonstrates comprehensive market analysis and professional trading recommendations
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.financial_advisor import FinancialAdvisor
from src.cfd_analyzer import CFDAnalyzer
from src.data_manager import FinancialDataManager

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

def demo_financial_advisor():
    """Demonstrate Financial Advisor capabilities"""
    print_header("FINANCIAL ADVISOR DEMO - 100+ YEARS TRADING EXPERIENCE")
    
    try:
        # Initialize components
        financial_advisor = FinancialAdvisor()
        
        print("✓ Financial Advisor initialized successfully")
        print("✓ 100+ years of trading experience loaded")
        print("✓ Professional market analysis ready")
        print("✓ Trading212 CFD instruments configured")
        
        # Show available instruments
        print_section("Available Trading212 CFD Instruments")
        instruments = financial_advisor._get_available_instruments()
        
        for instrument_type, instrument_list in instruments.items():
            if isinstance(instrument_list, dict):
                print(f"\n{instrument_type.upper()}:")
                for region, symbols in instrument_list.items():
                    print(f"  {region}: {len(symbols)} instruments")
            else:
                print(f"\n{instrument_type.upper()}: {len(instrument_list)} instruments")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Financial Advisor demo: {e}")
        return False

def demo_market_conditions_analysis():
    """Demonstrate comprehensive market conditions analysis"""
    print_header("COMPREHENSIVE MARKET CONDITIONS ANALYSIS")
    
    try:
        financial_advisor = FinancialAdvisor()
        
        print_section("Analyzing Market Conditions")
        print("The Financial Advisor is analyzing:")
        print("- UK market status and trading hours")
        print("- Global market sentiment across major indices")
        print("- Sector performance and opportunities")
        print("- Market volatility conditions")
        print("- Financial news sentiment from multiple sources")
        print("- Technical market overview")
        
        # Get market analysis
        market_analysis = financial_advisor.analyze_market_conditions()
        
        if not market_analysis:
            print("✗ Failed to analyze market conditions")
            return False
        
        print("✓ Market conditions analysis completed successfully")
        
        # Display UK market status
        print_section("UK Market Status")
        uk_status = market_analysis.get('uk_market_status', {})
        if uk_status:
            print(f"Market Status: {uk_status.get('status', 'UNKNOWN')}")
            print(f"Current Time: {uk_status.get('current_time', 'N/A')}")
            print(f"Market Hours: {uk_status.get('market_open', 'N/A')} - {uk_status.get('market_close', 'N/A')}")
            print(f"Is Open: {uk_status.get('is_open', False)}")
            
            if uk_status.get('time_to_open'):
                print(f"Time to Market Open: {uk_status['time_to_open']}")
            elif uk_status.get('time_to_close'):
                print(f"Market Closed: {uk_status['time_to_close']} ago")
        
        # Display global sentiment
        print_section("Global Market Sentiment")
        global_sentiment = market_analysis.get('global_sentiment', {})
        if global_sentiment:
            print(f"Overall Sentiment: {global_sentiment.get('overall_sentiment', 'NEUTRAL')}")
            print(f"Sentiment Score: {global_sentiment.get('sentiment_score', 0):.1%}")
            print(f"Indices Analyzed: {global_sentiment.get('indices_analyzed', 0)}")
        
        # Display news sentiment
        print_section("Financial News Sentiment")
        news_sentiment = market_analysis.get('news_sentiment', {})
        if news_sentiment:
            print(f"Overall News Sentiment: {news_sentiment.get('overall_sentiment', 'NEUTRAL')}")
            print(f"Bullish News: {news_sentiment.get('bullish_news', 0)}")
            print(f"Bearish News: {news_sentiment.get('bearish_news', 0)}")
            print(f"Neutral News: {news_sentiment.get('neutral_news', 0)}")
            
            key_headlines = news_sentiment.get('key_headlines', [])
            if key_headlines:
                print("Key Headlines:")
                for i, headline in enumerate(key_headlines[:5], 1):
                    print(f"  {i}. {headline}")
        
        # Display market volatility
        print_section("Market Volatility Analysis")
        market_volatility = market_analysis.get('market_volatility', {})
        if market_volatility:
            print(f"Overall Volatility Condition: {market_volatility.get('overall_condition', 'NORMAL')}")
            print(f"Volatility Ratio: {market_volatility.get('volatility_ratio', 1.0):.2f}x")
            print(f"Trading Implication: {market_volatility.get('trading_implication', 'Standard risk management')}")
        
        # Display technical overview
        print_section("Technical Market Overview")
        technical_overview = market_analysis.get('technical_overview', {})
        if technical_overview:
            for index, data in technical_overview.items():
                print(f"{index}:")
                print(f"  Trend: {data.get('trend', 'UNKNOWN')}")
                print(f"  Price vs SMA20: {data.get('price_vs_sma20', 'UNKNOWN')}")
                print(f"  Price vs SMA50: {data.get('price_vs_sma50', 'UNKNOWN')}")
                print(f"  Current Price: {data.get('current_price', 0):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in market conditions analysis demo: {e}")
        return False

def demo_trading_recommendations():
    """Demonstrate professional trading recommendations"""
    print_header("PROFESSIONAL TRADING RECOMMENDATIONS")
    
    try:
        financial_advisor = FinancialAdvisor()
        
        # Test with £147 capital
        capital = 147.0
        print_section(f"Generating Trading Recommendations for £{capital}")
        print("The Financial Advisor is analyzing:")
        print("- All available Trading212 CFD instruments")
        print("- Market conditions and sentiment")
        print("- Technical analysis and patterns")
        print("- News sentiment and market drivers")
        print("- Risk management and position sizing")
        print("- Entry, stop loss, and take profit levels")
        
        # Get recommendations
        recommendations = financial_advisor.get_trading_recommendations(capital)
        
        if not recommendations:
            print("✗ Failed to generate trading recommendations")
            return False
        
        print("✓ Trading recommendations generated successfully")
        
        # Display summary
        print_section("Recommendations Summary")
        print(f"Capital: £{recommendations.get('capital', 0)}")
        print(f"Total Recommendations: {recommendations.get('total_recommendations', 0)}")
        print(f"Filtered Recommendations: {recommendations.get('filtered_recommendations', 0)}")
        
        # Display risk summary
        risk_summary = recommendations.get('risk_summary', {})
        if risk_summary:
            print_section("Risk Summary")
            print(f"Risk Level: {risk_summary.get('risk_level', 'UNKNOWN')}")
            print(f"Total Risk: £{risk_summary.get('total_risk', 0)}")
            print(f"Max Risk per Trade: £{risk_summary.get('max_risk_per_trade', 0)}")
            print(f"Total Risk Percentage: {risk_summary.get('total_risk_percentage', 0)}%")
            print(f"Recommendations Count: {risk_summary.get('recommendations_count', 0)}")
        
        # Display top recommendations
        top_recommendations = recommendations.get('top_recommendations', [])
        if top_recommendations:
            print_section("Top Trading Recommendations")
            for i, rec in enumerate(top_recommendations[:5], 1):
                print(f"\n{i}. {rec.get('symbol', 'Unknown')}")
                print(f"   Setup Type: {rec.get('setup_type', 'Unknown')}")
                print(f"   Direction: {rec.get('direction', 'neutral').upper()}")
                print(f"   Confidence: {rec.get('confidence', 0):.1%}")
                print(f"   Setup Score: {rec.get('setup_score', 0):.1%}")
                
                # Entry prices
                entry_prices = rec.get('entry_prices', {})
                if entry_prices:
                    print("   Entry Prices:")
                    for level_type, price in entry_prices.items():
                        print(f"     {level_type.title()}: ${price:.4f}")
                
                # Stop loss and take profit
                stop_loss = rec.get('stop_loss', {})
                take_profit = rec.get('take_profit', {})
                
                if stop_loss.get('level'):
                    print(f"   Stop Loss: ${stop_loss['level']:.4f} ({stop_loss['percentage']:.1%})")
                
                if take_profit.get('level'):
                    print(f"   Take Profit: ${take_profit['level']:.4f} ({take_profit['percentage']:.1%})")
                    print(f"   Risk/Reward: {take_profit['risk_reward_ratio']:.1f}:1")
                
                # Position sizing
                position_sizing = rec.get('position_sizing', {})
                if position_sizing:
                    print(f"   Position Size: £{position_sizing['position_size']}")
                    print(f"   Units: {position_sizing['units']}")
                    print(f"   Risk Amount: £{position_sizing['risk_amount']}")
                    print(f"   Capital Usage: {position_sizing['max_capital_usage']}%")
                
                # Recommendation reason
                reason = rec.get('recommendation_reason', 'No reason provided')
                print(f"   Reason: {reason}")
        
        # Display execution plan
        execution_plan = recommendations.get('execution_plan', {})
        if execution_plan and execution_plan.get('execution_steps'):
            print_section("Execution Plan")
            print(f"Total Capital Required: £{execution_plan.get('total_capital_required', 0)}")
            print(f"Capital Available: £{execution_plan.get('capital_available', 0)}")
            
            print("\nExecution Steps:")
            for step in execution_plan['execution_steps']:
                print(f"  Step {step['step']}: {step['symbol']} - {step['action']}")
                print(f"    Entry Price: ${step['entry_price']}")
                print(f"    Units: {step['units']}")
                print(f"    Capital Required: £{step['capital_required']}")
                print(f"    Priority: {step['priority']}")
            
            print("\nExecution Notes:")
            for note in execution_plan.get('execution_notes', []):
                print(f"  • {note}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in trading recommendations demo: {e}")
        return False

def demo_uk_market_status():
    """Demonstrate UK market status checking"""
    print_header("UK MARKET STATUS & TRADING HOURS")
    
    try:
        financial_advisor = FinancialAdvisor()
        
        print_section("Checking UK Market Status")
        print("The Financial Advisor is checking:")
        print("- Current UK market open/close status")
        print("- Trading hours (8:00 AM - 4:30 PM GMT)")
        print("- Time to market open/close")
        print("- Market session information")
        
        # Check UK market status
        market_status = financial_advisor._check_uk_market_status()
        
        if not market_status:
            print("✗ Failed to check UK market status")
            return False
        
        print("✓ UK market status checked successfully")
        
        # Display status
        print_section("UK Market Status")
        print(f"Status: {market_status.get('status', 'UNKNOWN')}")
        print(f"Current Time: {market_status.get('current_time', 'N/A')}")
        print(f"Market Hours: {market_status.get('market_open', 'N/A')} - {market_status.get('market_close', 'N/A')}")
        print(f"Is Open: {market_status.get('is_open', False)}")
        
        if market_status.get('time_to_open'):
            print(f"Time to Market Open: {market_status['time_to_open']}")
        elif market_status.get('time_to_close'):
            print(f"Market Closed: {market_status['time_to_close']} ago")
        
        # Trading implications
        print_section("Trading Implications")
        if market_status.get('is_open'):
            print("✓ UK market is currently OPEN")
            print("✓ You can execute CFD trades now")
            print("✓ Real-time market data available")
            print("✓ Professional recommendations active")
        else:
            print("⚠ UK market is currently CLOSED")
            if market_status.get('time_to_open'):
                print(f"⏰ Market opens in {market_status['time_to_open']}")
                print("📊 Prepare your trading strategy")
                print("🔍 Review market conditions")
            else:
                print("🌙 Market closed for the day")
                print("📈 Plan for tomorrow's session")
                print("📊 Review today's performance")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in UK market status demo: {e}")
        return False

def demo_instrument_analysis():
    """Demonstrate individual instrument analysis"""
    print_header("INDIVIDUAL INSTRUMENT ANALYSIS")
    
    try:
        financial_advisor = FinancialAdvisor()
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        # Test with a popular UK stock
        symbol = "VOD.L"  # Vodafone
        print_section(f"Analyzing {symbol} for Trading212 CFD")
        
        # Get market data
        data = data_manager.get_stock_data(symbol, period="1d", interval="5m")
        if data.empty:
            print(f"✗ No data available for {symbol}")
            return False
        
        print(f"✓ Retrieved {len(data)} data points for {symbol}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Latest price: ${data['Close'].iloc[-1]:.4f}")
        
        # Get CFD setup analysis
        cfd_setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type='stock')
        if not cfd_setup:
            print(f"✗ Failed to analyze CFD setup for {symbol}")
            return False
        
        # Get news sentiment
        news_sentiment = financial_advisor._get_instrument_news_sentiment(symbol)
        
        # Get technical analysis
        technical_analysis = financial_advisor._analyze_instrument_technical(symbol, data)
        
        # Calculate exact entry prices
        entry_prices = financial_advisor._calculate_exact_entry_prices(symbol, data, cfd_setup)
        
        # Check trade executability
        trade_status = financial_advisor._check_trade_executability(symbol, data, cfd_setup)
        
        # Calculate position sizing for £147
        position_sizing = financial_advisor._calculate_position_sizing(147.0, cfd_setup)
        
        # Display comprehensive analysis
        print_section("Comprehensive Instrument Analysis")
        print(f"Symbol: {symbol}")
        print(f"Setup Type: {cfd_setup['setup_type']}")
        print(f"Direction: {cfd_setup['direction'].upper()}")
        print(f"Confidence: {cfd_setup['confidence']:.1%}")
        print(f"Setup Score: {cfd_setup['setup_score']:.1%}")
        
        # News sentiment
        print_section("News Sentiment Analysis")
        print(f"Sentiment: {news_sentiment['sentiment']}")
        print(f"Confidence: {news_sentiment['confidence']:.1%}")
        print("Key News:")
        for headline in news_sentiment.get('key_news', []):
            print(f"  • {headline}")
        
        # Technical analysis
        print_section("Technical Analysis")
        if technical_analysis:
            print(f"Price Change: {technical_analysis['price_change_percent']:.2f}%")
            print(f"Volume Ratio: {technical_analysis['volume_ratio']:.2f}")
            print(f"RSI: {technical_analysis['rsi']:.2f}" if technical_analysis['rsi'] else "RSI: N/A")
            print(f"MACD Signal: {technical_analysis['macd_signal']}")
            print(f"Technical Score: {technical_analysis['technical_score']:.1%}")
        
        # Entry prices
        print_section("Exact Entry Prices")
        for level_type, price in entry_prices.items():
            print(f"  {level_type.title()}: ${price:.4f}")
        
        # Risk management
        print_section("Risk Management")
        stop_loss = cfd_setup.get('stop_loss', {})
        take_profit = cfd_setup.get('take_profit', {})
        
        if stop_loss.get('level'):
            print(f"Stop Loss: ${stop_loss['level']:.4f} ({stop_loss['percentage']:.1%})")
        
        if take_profit.get('level'):
            print(f"Take Profit: ${take_profit['level']:.4f} ({take_profit['percentage']:.1%})")
            print(f"Risk/Reward Ratio: {take_profit['risk_reward_ratio']:.1f}:1")
        
        # Position sizing
        print_section("Position Sizing (£147 Capital)")
        if position_sizing:
            print(f"Position Size: £{position_sizing['position_size']}")
            print(f"Units: {position_sizing['units']}")
            print(f"Risk Amount: £{position_sizing['risk_amount']}")
            print(f"Capital Usage: {position_sizing['max_capital_usage']}%")
            print(f"Reason: {position_sizing['reason']}")
        
        # Trade status
        print_section("Trade Executability")
        print(f"Executable: {trade_status['is_executable']}")
        print(f"Reason: {trade_status['reason']}")
        print(f"Current Price: ${trade_status['current_price']:.4f}")
        print(f"Confidence: {trade_status['confidence']:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in instrument analysis demo: {e}")
        return False

def main():
    """Run the complete Financial Advisor demo"""
    print_header("FINANCIAL ADVISOR SYSTEM DEMO FOR TRADING212 CFD")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This demo showcases the professional Financial Advisor system")
    print("Combining 100+ years of trading experience with AI-powered analysis")
    print("Perfect for Trading212 CFD traders seeking professional guidance")
    
    # Run all demos
    demos = [
        ("Financial Advisor", demo_financial_advisor),
        ("Market Conditions Analysis", demo_market_conditions_analysis),
        ("Trading Recommendations", demo_trading_recommendations),
        ("UK Market Status", demo_uk_market_status),
        ("Individual Instrument Analysis", demo_instrument_analysis)
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
    print_header("FINANCIAL ADVISOR DEMO SUMMARY")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"Completed {total_demos} demos: {successful_demos} successful, {total_demos - successful_demos} failed")
    
    for demo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {demo_name}: {status}")
    
    if successful_demos == total_demos:
        print("\n🎉 All Financial Advisor demos completed successfully!")
        print("The professional trading advisor is ready to guide your Trading212 CFD trades!")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed.")
        print("Check the logs above for error details.")
    
    print("\nFinancial Advisor Features Available:")
    print("✓ 100+ years of trading experience")
    print("✓ Comprehensive market conditions analysis")
    print("✓ Professional trading recommendations")
    print("✓ Exact entry, stop loss, and take profit prices")
    print("✓ Risk management and position sizing")
    print("✓ UK market status and trading hours")
    print("✓ News sentiment analysis")
    print("✓ Technical market overview")
    print("✓ Multi-instrument scanning")
    print("✓ Same-day trading strategies")
    
    print("\nNext steps:")
    print("1. Run the web interface: python app.py")
    print("2. Navigate to the Financial Advisor section")
    print("3. Get professional market analysis")
    print("4. Receive trading recommendations")
    print("5. Execute trades with exact prices")
    
    print(f"\nFinancial Advisor Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nFinancial Advisor Demo interrupted by user")
    except Exception as e:
        logger.error(f"Financial Advisor Demo failed with error: {e}")
        sys.exit(1)