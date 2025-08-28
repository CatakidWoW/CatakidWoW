"""
Financial AI Web Application
Main Flask application for the Financial AI System with CFD Analysis
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import json

from src.data_manager import FinancialDataManager
from src.ai_models import FinancialAIModels
from src.portfolio_manager import PortfolioManager
from src.trading_strategies import TradingStrategies
from src.cfd_analyzer import CFDAnalyzer
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/financial_ai.log'),
        logging.StreamHandler()
    ]
)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

app = Flask(__name__)
CORS(app)

# Initialize components
config = Config()
data_manager = FinancialDataManager()
ai_models = FinancialAIModels()
portfolio_manager = PortfolioManager()
trading_strategies = TradingStrategies()
cfd_analyzer = CFDAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/stocks/<symbol>')
def get_stock_data(symbol):
    """Get stock data for a specific symbol"""
    try:
        period = request.args.get('period', '1y')
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Convert to JSON-serializable format
        data_json = data.reset_index()
        data_json['Date'] = data_json.index.astype(str)
        
        return jsonify({
            'symbol': symbol,
            'data': data_json.to_dict('records'),
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/<symbol>/info')
def get_stock_info(symbol):
    """Get company information for a specific symbol"""
    try:
        info = data_manager.get_company_info(symbol)
        
        if not info:
            return jsonify({'error': f'No information found for {symbol}'}), 404
        
        return jsonify(info)
        
    except Exception as e:
        app.logger.error(f"Error fetching info for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/create', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        strategy = data.get('strategy', 'equal_weight')
        initial_capital = data.get('initial_capital', config.DEFAULT_INITIAL_CAPITAL)
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        # Create portfolio
        portfolio = portfolio_manager.create_portfolio(symbols, strategy=strategy)
        
        if not portfolio:
            return jsonify({'error': 'Failed to create portfolio'}), 500
        
        return jsonify({
            'message': 'Portfolio created successfully',
            'portfolio': portfolio
        })
        
    except Exception as e:
        app.logger.error(f"Error creating portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/summary')
def get_portfolio_summary():
    """Get current portfolio summary"""
    try:
        summary = portfolio_manager.get_portfolio_summary()
        
        if not summary:
            return jsonify({'error': 'No portfolio exists'}), 404
        
        return jsonify(summary)
        
    except Exception as e:
        app.logger.error(f"Error getting portfolio summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio weights"""
    try:
        data = request.get_json()
        method = data.get('method', 'sharpe_ratio')
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        # Get historical data for optimization
        stock_data = data_manager.get_multiple_stocks(symbols, period='1y')
        
        if not stock_data:
            return jsonify({'error': 'No data available for optimization'}), 404
        
        # Calculate returns
        returns_data = pd.DataFrame()
        for symbol, data in stock_data.items():
            if not data.empty and 'Returns' in data.columns:
                returns_data[symbol] = data['Returns']
        
        if returns_data.empty:
            return jsonify({'error': 'No returns data available'}), 404
        
        # Optimize portfolio
        result = portfolio_manager.optimize_portfolio(returns_data, method=method)
        
        if not result.get('success', False):
            return jsonify({'error': result.get('message', 'Optimization failed')}), 500
        
        return jsonify({
            'message': 'Portfolio optimized successfully',
            'result': result
        })
        
    except Exception as e:
        app.logger.error(f"Error optimizing portfolio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/signals/<symbol>')
def get_trading_signals(symbol):
    """Get trading signals for a specific symbol"""
    try:
        strategy = request.args.get('strategy', 'combined')
        period = request.args.get('period', '1y')
        
        # Get stock data
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Generate signals
        signals = trading_strategies.generate_signals(data, strategy=strategy)
        
        if signals.empty:
            return jsonify({'error': 'Failed to generate signals'}), 500
        
        # Get latest signals
        latest_signals = signals.iloc[-1].to_dict()
        
        return jsonify({
            'symbol': symbol,
            'strategy': strategy,
            'signals': latest_signals,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error generating signals for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading/recommendations/<symbol>')
def get_trading_recommendations(symbol):
    """Get trading recommendations for a specific symbol"""
    try:
        strategy = request.args.get('strategy', 'combined')
        period = request.args.get('period', '1y')
        
        # Get stock data
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Get recommendations
        recommendations = trading_strategies.get_trading_recommendations(data, strategy=strategy)
        
        if not recommendations:
            return jsonify({'error': 'Failed to generate recommendations'}), 500
        
        return jsonify(recommendations)
        
    except Exception as e:
        app.logger.error(f"Error getting recommendations for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train', methods=['POST'])
def train_ai_models():
    """Train AI models for a specific symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        model_type = data.get('model_type', 'all')  # 'ml', 'lstm', 'cnn', 'all'
        
        if not symbol:
            return jsonify({'error': 'No symbol provided'}), 400
        
        # Get training data
        training_data = data_manager.get_stock_data(symbol, period='2y')
        
        if training_data.empty:
            return jsonify({'error': f'No training data available for {symbol}'}), 404
        
        results = {}
        
        # Train models based on type
        if model_type in ['ml', 'all']:
            ml_results = ai_models.train_ml_models(training_data)
            results['ml_models'] = ml_results
        
        if model_type in ['lstm', 'all']:
            lstm_results = ai_models.train_lstm_model(training_data)
            results['lstm'] = lstm_results
        
        if model_type in ['cnn', 'all']:
            cnn_results = ai_models.train_cnn_model(training_data)
            results['cnn'] = cnn_results
        
        # Save models
        ai_models.save_models(f'{symbol}_models')
        
        return jsonify({
            'message': 'Models trained successfully',
            'results': results
        })
        
    except Exception as e:
        app.logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/predict/<symbol>')
def get_ai_predictions(symbol):
    """Get AI predictions for a specific symbol"""
    try:
        model_name = request.args.get('model', 'ensemble')
        period = request.args.get('period', '1y')
        
        # Get prediction data
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Load models if not already loaded
        if not ai_models.models:
            ai_models.load_models(f'{symbol}_models')
        
        # Make predictions
        if model_name == 'ensemble':
            predictions = ai_models.ensemble_predict(data)
        else:
            predictions = ai_models.predict(model_name, data)
        
        if len(predictions) == 0:
            return jsonify({'error': 'Failed to generate predictions'}), 500
        
        # Get latest prediction
        latest_prediction = predictions[-1]
        current_price = data['Close'].iloc[-1]
        
        prediction_result = {
            'symbol': symbol,
            'model': model_name,
            'current_price': current_price,
            'predicted_price': float(latest_prediction),
            'predicted_change': float((latest_prediction - current_price) / current_price),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(prediction_result)
        
    except Exception as e:
        app.logger.error(f"Error getting predictions for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/<symbol>')
def backtest_strategy(symbol):
    """Backtest a trading strategy for a specific symbol"""
    try:
        strategy = request.args.get('strategy', 'combined')
        period = request.args.get('period', '1y')
        initial_capital = float(request.args.get('initial_capital', 100000))
        
        # Get backtest data
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Run backtest
        results = trading_strategies.backtest_strategy(
            data, strategy=strategy, initial_capital=initial_capital
        )
        
        if not results:
            return jsonify({'error': 'Backtest failed'}), 500
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Error running backtest for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/overview')
def get_market_overview():
    """Get market overview data"""
    try:
        # Get major indices
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        market_data = data_manager.get_market_data(indices)
        
        market_overview = {}
        for index, data in market_data.items():
            if not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                
                market_overview[index] = {
                    'name': _get_index_name(index),
                    'current': float(latest['Close']),
                    'change': float(latest['Close'] - prev['Close']),
                    'change_pct': float((latest['Close'] - prev['Close']) / prev['Close']),
                    'volume': int(latest['Volume'])
                }
        
        return jsonify({
            'market_overview': market_overview,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error getting market overview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cfd/setups/<symbol>')
def get_cfd_setup(symbol):
    """Get CFD setup analysis for a specific symbol"""
    try:
        period = request.args.get('period', '1y')
        
        # Get stock data
        data = data_manager.get_stock_data(symbol, period=period)
        
        if data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Analyze CFD setup
        setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type='stock')
        
        if not setup:
            return jsonify({'error': 'Failed to analyze CFD setup'}), 500
        
        return jsonify({
            'symbol': symbol,
            'setup': setup,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error analyzing CFD setup for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cfd/top-setups')
def get_top_cfd_setups():
    """Get top CFD setups across multiple instruments"""
    try:
        instrument_type = request.args.get('type', 'stocks')  # stocks, indices, forex, commodities, crypto
        region = request.args.get('region', 'US')  # US, UK, EU, AU
        min_confidence = float(request.args.get('min_confidence', 0.6))
        max_instruments = int(request.args.get('max_instruments', 20))
        
        # Get instrument list
        if instrument_type == 'stocks' and region in cfd_analyzer.cfd_instruments['stocks']:
            symbols = cfd_analyzer.cfd_instruments['stocks'][region][:max_instruments]
        elif instrument_type in cfd_analyzer.cfd_instruments:
            symbols = cfd_analyzer.cfd_instruments[instrument_type][:max_instruments]
        else:
            return jsonify({'error': f'Invalid instrument type: {instrument_type}'}), 400
        
        # Get data for all instruments
        instruments_data = {}
        for symbol in symbols:
            try:
                data = data_manager.get_stock_data(symbol, period='6mo')
                if not data.empty:
                    instruments_data[symbol] = data
            except Exception as e:
                app.logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                continue
        
        if not instruments_data:
            return jsonify({'error': 'No data available for analysis'}), 404
        
        # Get top CFD setups
        top_setups = cfd_analyzer.get_top_cfd_setups(instruments_data, min_confidence)
        
        return jsonify({
            'instrument_type': instrument_type,
            'region': region,
            'total_instruments': len(instruments_data),
            'setups_found': len(top_setups),
            'top_setups': top_setups[:10],  # Return top 10
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error getting top CFD setups: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cfd/report')
def generate_cfd_report():
    """Generate comprehensive CFD trading report"""
    try:
        instrument_type = request.args.get('type', 'stocks')
        region = request.args.get('region', 'US')
        min_confidence = float(request.args.get('min_confidence', 0.6))
        
        # Get instrument list
        if instrument_type == 'stocks' and region in cfd_analyzer.cfd_instruments['stocks']:
            symbols = cfd_analyzer.cfd_instruments['stocks'][region]
        elif instrument_type in cfd_analyzer.cfd_instruments:
            symbols = cfd_analyzer.cfd_instruments[instrument_type]
        else:
            return jsonify({'error': f'Invalid instrument type: {instrument_type}'}), 400
        
        # Get data for analysis
        instruments_data = {}
        for symbol in symbols[:30]:  # Limit to 30 instruments for performance
            try:
                data = data_manager.get_stock_data(symbol, period='6mo')
                if not data.empty:
                    instruments_data[symbol] = data
            except Exception as e:
                app.logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                continue
        
        if not instruments_data:
            return jsonify({'error': 'No data available for report generation'}), 404
        
        # Get top setups
        top_setups = cfd_analyzer.get_top_cfd_setups(instruments_data, min_confidence)
        
        # Generate report
        report = cfd_analyzer.generate_cfd_report(top_setups)
        
        return jsonify({
            'report': report,
            'summary': {
                'total_instruments': len(instruments_data),
                'setups_found': len(top_setups),
                'top_confidence': top_setups[0].get('confidence', 0) if top_setups else 0,
                'average_confidence': sum(s.get('confidence', 0) for s in top_setups) / len(top_setups) if top_setups else 0
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error generating CFD report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cfd/instruments')
def get_cfd_instruments():
    """Get available CFD instruments"""
    try:
        return jsonify({
            'instruments': cfd_analyzer.cfd_instruments,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error getting CFD instruments: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cfd/scan')
def scan_cfd_opportunities():
    """Scan for CFD trading opportunities"""
    try:
        scan_type = request.args.get('scan_type', 'all')  # all, momentum, breakout, reversal
        min_confidence = float(request.args.get('min_confidence', 0.7))
        
        opportunities = []
        
        # Scan different instrument types
        for instrument_type, instruments in cfd_analyzer.cfd_instruments.items():
            if scan_type != 'all' and instrument_type != scan_type:
                continue
                
            if isinstance(instruments, dict):
                # Handle regional stocks
                for region, symbols in instruments.items():
                    for symbol in symbols[:10]:  # Limit per region
                        try:
                            data = data_manager.get_stock_data(symbol, period='1mo')
                            if not data.empty:
                                setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type)
                                if setup and setup.get('confidence', 0) >= min_confidence:
                                    setup['symbol'] = symbol
                                    setup['region'] = region
                                    setup['instrument_type'] = instrument_type
                                    opportunities.append(setup)
                        except Exception as e:
                            app.logger.warning(f"Failed to scan {symbol}: {str(e)}")
                            continue
            else:
                # Handle other instrument types
                for symbol in instruments[:10]:
                    try:
                        data = data_manager.get_stock_data(symbol, period='1mo')
                        if not data.empty:
                            setup = cfd_analyzer.analyze_cfd_setups(data, instrument_type)
                            if setup and setup.get('confidence', 0) >= min_confidence:
                                setup['symbol'] = symbol
                                setup['instrument_type'] = instrument_type
                                opportunities.append(setup)
                    except Exception as e:
                        app.logger.warning(f"Failed to scan {symbol}: {str(e)}")
                        continue
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return jsonify({
            'scan_type': scan_type,
            'opportunities_found': len(opportunities),
            'top_opportunities': opportunities[:20],  # Return top 20
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error scanning CFD opportunities: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _get_index_name(symbol):
    """Get human-readable name for market indices"""
    names = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones Industrial Average',
        '^IXIC': 'NASDAQ Composite',
        '^RUT': 'Russell 2000'
    }
    return names.get(symbol, symbol)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)