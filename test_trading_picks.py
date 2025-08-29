#!/usr/bin/env python3
"""
Test Trading Picks Generation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from financial_advisor import FinancialAdvisor
from real_time_data import RealTimeDataManager

def test_trading_picks():
    """Test the trading picks generation"""
    print('🚀 GENERATING LIVE TRADING PICKS...')
    print('=' * 60)
    
    try:
        advisor = FinancialAdvisor()
        picks = advisor.get_trading_picks(capital=147.0)
        
        if picks:
            print(f'✅ Generated {len(picks)} trading picks!')
            print()
            
            for i, pick in enumerate(picks[:3], 1):  # Show first 3 picks
                print(f'🎯 PICK #{i}: {pick["instrument"]}')
                print(f'   Setup Type: {pick["setup_type"]}')
                print(f'   Trigger: {pick["trigger_description"]}')
                print(f'   Stop Guide: {pick["stop_guide"]}')
                print(f'   Target Guide: {pick["target_guide"]}')
                print(f'   Likelihood: {pick["likelihood_text"]}')
                print(f'   Notes: {pick["notes"]}')
                print('-' * 40)
        else:
            print('❌ No trading picks generated')
            print('🔍 Checking what data is available...')
            
            # Test individual data retrieval
            rtdm = RealTimeDataManager()
            
            test_symbols = ['AAPL', 'MSFT', 'TSLA']
            for symbol in test_symbols:
                try:
                    data = rtdm.get_live_market_data(symbol, '1h', '1d')
                    if not data.empty:
                        print(f'✅ {symbol}: {len(data)} data points, Latest: ${data["Close"].iloc[-1]:.2f}')
                    else:
                        print(f'❌ {symbol}: No data')
                except Exception as e:
                    print(f'❌ {symbol}: Error - {str(e)}')
                    
    except Exception as e:
        print(f'❌ Error generating trading picks: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trading_picks()