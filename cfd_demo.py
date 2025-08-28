#!/usr/bin/env python3
"""
CFD Analysis Demo for Trading212
Demonstrates the specialized CFD trading setup analyzer
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

def demo_cfd_analyzer():
    """Demonstrate CFD analyzer capabilities"""
    print_header("CFD ANALYZER DEMO FOR TRADING212")
    
    try:
        # Initialize components
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        print("✓ CFD Analyzer initialized successfully")
        print("✓ Data Manager initialized successfully")
        
        # Show available instruments
        print_section("Available CFD Instruments")
        print("The system supports the following instrument types:")
        
        for instrument_type, instruments in cfd_analyzer.cfd_instruments.items():
            if isinstance(instruments, dict):
                print(f"\n{instrument_type.upper()}:")
                for region, symbols in instruments.items():
                    print(f"  {region}: {len(symbols)} instruments")
            else:
                print(f"\n{instrument_type.upper()}: {len(instruments)} instruments")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CFD analyzer demo: {e}")
        return False

def demo_individual_cfd_setup():
    """Demonstrate individual CFD setup analysis"""
    print_header("INDIVIDUAL CFD SETUP ANALYSIS")
    
    try:
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        # Analyze a popular stock
        symbol = "AAPL"
        print_section(f"Analyzing CFD Setup for {symbol}")
        
        # Get data
        data = data_manager.get_stock_data(symbol, period="6mo")
        if data.empty:
            print(f"✗ No data available for {symbol}")
            return False
        
        print(f"✓ Retrieved {len(data)} data points for {symbol}")
        print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  Latest price: ${data['Close'].iloc[-1]:.2f}")
        
        # Analyze CFD setup
        setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type='stock')
        
        if not setup:
            print(f"✗ Failed to analyze CFD setup for {symbol}")
            return False
        
        # Display results
        print_section("CFD Setup Analysis Results")
        print(f"Setup Type: {setup['setup_type']}")
        print(f"Direction: {setup['direction'].upper()}")
        print(f"Confidence: {setup['confidence']:.1%}")
        print(f"Setup Score: {setup['setup_score']:.1%}")
        
        # Trend analysis
        print_section("Trend Analysis")
        trend = setup['trend_analysis']
        print(f"Trend Score: {trend['trend_score']:.1%}")
        print(f"Trend Strength: {trend['trend_strength']}")
        print(f"Trend Direction: {trend['trend_direction']}")
        
        # Momentum analysis
        print_section("Momentum Analysis")
        momentum = setup['momentum_analysis']
        print(f"Momentum Score: {momentum['momentum_score']:.1%}")
        print(f"RSI: {momentum['rsi']:.2f}" if momentum['rsi'] else "RSI: N/A")
        print(f"MACD Signal: {momentum['macd_signal']}")
        
        # Entry levels
        print_section("Entry Levels")
        entry_levels = setup['entry_levels']
        for level_type, price in entry_levels.items():
            print(f"  {level_type.title()}: ${price:.4f}")
        
        # Risk management
        print_section("Risk Management")
        stop_loss = setup['stop_loss']
        take_profit = setup['take_profit']
        
        if stop_loss['level']:
            print(f"Stop Loss: ${stop_loss['level']:.4f} ({stop_loss['percentage']:.1%})")
        
        if take_profit['level']:
            print(f"Take Profit: ${take_profit['level']:.4f} ({take_profit['percentage']:.1%})")
            print(f"Risk/Reward Ratio: {take_profit['risk_reward_ratio']:.1f}:1")
        
        # Risk metrics
        print_section("Risk Metrics")
        risk_metrics = setup['risk_metrics']
        if risk_metrics:
            print(f"Position Size: {risk_metrics['position_size_percentage']:.1f}% of account")
            print(f"Max Loss: {risk_metrics['max_loss_percentage']:.1f}% per trade")
            print(f"Risk Amount: ${risk_metrics['risk_amount']:.4f}")
            print(f"Reward Amount: ${risk_metrics['reward_amount']:.4f}")
        
        # Support and resistance
        print_section("Support & Resistance")
        sr = setup['support_resistance']
        if sr['support_levels']:
            print(f"Support Levels: {', '.join([f'${level:.4f}' for level in sr['support_levels'][:3]])}")
        if sr['resistance_levels']:
            print(f"Resistance Levels: {', '.join([f'${level:.4f}' for level in sr['resistance_levels'][:3]])}")
        print(f"Current Position: {sr['current_position']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in individual CFD setup demo: {e}")
        return False

def demo_top_cfd_setups():
    """Demonstrate top CFD setups analysis"""
    print_header("TOP CFD SETUPS ANALYSIS")
    
    try:
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        # Get top setups for US stocks
        print_section("Analyzing Top US Stock CFD Setups")
        
        us_stocks = cfd_analyzer.cfd_instruments['stocks']['US'][:10]  # Top 10 US stocks
        instruments_data = {}
        
        print(f"Analyzing {len(us_stocks)} US stocks...")
        
        for symbol in us_stocks:
            try:
                data = data_manager.get_stock_data(symbol, period="6mo")
                if not data.empty:
                    instruments_data[symbol] = data
                    print(f"  ✓ {symbol}: {len(data)} data points")
                else:
                    print(f"  ✗ {symbol}: No data available")
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {str(e)}")
                continue
        
        if not instruments_data:
            print("✗ No data available for analysis")
            return False
        
        # Get top setups
        print_section("Finding Top CFD Setups")
        top_setups = cfd_analyzer.get_top_cfd_setups(instruments_data, min_confidence=0.6)
        
        if not top_setups:
            print("✗ No CFD setups found matching criteria")
            return False
        
        print(f"✓ Found {len(top_setups)} CFD setups with confidence ≥ 60%")
        
        # Display top 5 setups
        print_section("Top 5 CFD Setups")
        for i, setup in enumerate(top_setups[:5], 1):
            print(f"\n{i}. {setup['symbol']}")
            print(f"   Setup Type: {setup['setup_type']}")
            print(f"   Direction: {setup['direction'].upper()}")
            print(f"   Confidence: {setup['confidence']:.1%}")
            print(f"   Setup Score: {setup['setup_score']:.1%}")
            
            # Show entry levels
            if setup['entry_levels']:
                entry = setup['entry_levels'].get('moderate', setup['entry_levels'].get('current', 0))
                print(f"   Entry Level: ${entry:.4f}")
            
            # Show risk/reward
            if setup['take_profit'] and setup['stop_loss']:
                rr_ratio = setup['take_profit']['risk_reward_ratio']
                print(f"   Risk/Reward: {rr_ratio:.1f}:1")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in top CFD setups demo: {e}")
        return False

def demo_cfd_report_generation():
    """Demonstrate CFD report generation"""
    print_header("CFD REPORT GENERATION")
    
    try:
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        # Get data for report
        print_section("Gathering Data for Report")
        
        # Use a smaller set for demo
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        instruments_data = {}
        
        for symbol in symbols:
            try:
                data = data_manager.get_stock_data(symbol, period="6mo")
                if not data.empty:
                    instruments_data[symbol] = data
                    print(f"  ✓ {symbol}: {len(data)} data points")
                else:
                    print(f"  ✗ {symbol}: No data available")
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {str(e)}")
                continue
        
        if not instruments_data:
            print("✗ No data available for report generation")
            return False
        
        # Get setups
        print_section("Analyzing CFD Setups")
        setups = cfd_analyzer.get_top_cfd_setups(instruments_data, min_confidence=0.5)
        
        if not setups:
            print("✗ No CFD setups found")
            return False
        
        print(f"✓ Found {len(setups)} CFD setups")
        
        # Generate report
        print_section("Generating CFD Report")
        report = cfd_analyzer.generate_cfd_report(setups)
        
        print("✓ CFD Report generated successfully")
        print(f"Report length: {len(report)} characters")
        
        # Show report preview
        print_section("Report Preview (First 500 characters)")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CFD report generation demo: {e}")
        return False

def demo_cfd_opportunity_scanning():
    """Demonstrate CFD opportunity scanning"""
    print_header("CFD OPPORTUNITY SCANNING")
    
    try:
        cfd_analyzer = CFDAnalyzer()
        data_manager = FinancialDataManager()
        
        print_section("Scanning for CFD Trading Opportunities")
        print("This demo shows how the system scans multiple instruments for trading opportunities")
        
        # Scan different instrument types
        scan_results = {}
        
        for instrument_type, instruments in cfd_analyzer.cfd_instruments.items():
            print(f"\nScanning {instrument_type.upper()}...")
            
            if isinstance(instruments, dict):
                # Regional stocks
                for region, symbols in instruments.items():
                    region_opportunities = []
                    for symbol in symbols[:5]:  # Limit to 5 per region for demo
                        try:
                            data = data_manager.get_stock_data(symbol, period="1mo")
                            if not data.empty:
                                setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type)
                                if setup and setup.get('confidence', 0) >= 0.6:
                                    setup['symbol'] = symbol
                                    setup['region'] = region
                                    setup['instrument_type'] = instrument_type
                                    region_opportunities.append(setup)
                        except Exception as e:
                            continue
                    
                    if region_opportunities:
                        scan_results[f"{instrument_type}_{region}"] = region_opportunities
                        print(f"  {region}: {len(region_opportunities)} opportunities found")
                    else:
                        print(f"  {region}: No opportunities found")
            else:
                # Other instruments
                opportunities = []
                for symbol in instruments[:5]:  # Limit to 5 for demo
                    try:
                        data = data_manager.get_stock_data(symbol, period="1mo")
                        if not data.empty:
                            setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type)
                            if setup and setup.get('confidence', 0) >= 0.6:
                                setup['symbol'] = symbol
                                setup['instrument_type'] = instrument_type
                                opportunities.append(setup)
                    except Exception as e:
                        continue
                
                if opportunities:
                    scan_results[instrument_type] = opportunities
                    print(f"  {len(opportunities)} opportunities found")
                else:
                    print(f"  No opportunities found")
        
        # Summary
        print_section("Scanning Summary")
        total_opportunities = sum(len(opps) for opps in scan_results.values())
        print(f"Total opportunities found: {total_opportunities}")
        
        for category, opportunities in scan_results.items():
            if opportunities:
                best_confidence = max(opp.get('confidence', 0) for opp in opportunities)
                print(f"{category}: {len(opportunities)} opportunities (best: {best_confidence:.1%})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CFD opportunity scanning demo: {e}")
        return False

def main():
    """Run the complete CFD demo"""
    print_header("CFD ANALYSIS SYSTEM DEMO FOR TRADING212")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This demo showcases the specialized CFD trading setup analyzer")
    print("Perfect for Trading212 CFD traders looking for optimal entry points")
    
    # Run all demos
    demos = [
        ("CFD Analyzer", demo_cfd_analyzer),
        ("Individual CFD Setup", demo_individual_cfd_setup),
        ("Top CFD Setups", demo_top_cfd_setups),
        ("CFD Report Generation", demo_cfd_report_generation),
        ("CFD Opportunity Scanning", demo_cfd_opportunity_scanning)
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
    print_header("CFD DEMO SUMMARY")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"Completed {total_demos} demos: {successful_demos} successful, {total_demos - successful_demos} failed")
    
    for demo_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {demo_name}: {status}")
    
    if successful_demos == total_demos:
        print("\n🎉 All CFD demos completed successfully!")
        print("The CFD Analysis system is ready to find Trading212 trading opportunities!")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed.")
        print("Check the logs above for error details.")
    
    print("\nCFD Trading Features Available:")
    print("✓ Real-time CFD setup analysis")
    print("✓ Multi-instrument opportunity scanning")
    print("✓ Entry, stop loss, and take profit levels")
    print("✓ Risk management and position sizing")
    print("✓ Support and resistance identification")
    print("✓ Trend and momentum analysis")
    print("✓ Comprehensive trading reports")
    
    print("\nNext steps:")
    print("1. Run the web interface: python app.py")
    print("2. Navigate to the CFD Analysis section")
    print("3. Scan for trading opportunities")
    print("4. Analyze individual setups")
    print("5. Generate comprehensive reports")
    
    print(f"\nCFD Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCFD Demo interrupted by user")
    except Exception as e:
        logger.error(f"CFD Demo failed with error: {e}")
        sys.exit(1)