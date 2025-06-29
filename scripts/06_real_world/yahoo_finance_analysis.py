#!/usr/bin/env python3
"""
Financial Analysis with Yahoo Finance and DSPy

This script demonstrates how to build intelligent financial analysis tools using DSPy and Yahoo Finance data.
It covers stock analysis, portfolio optimization, and market sentiment analysis.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import dspy
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from utils import setup_default_lm, print_step, print_result, print_error
from dotenv import load_dotenv

def main():
    """Main function demonstrating financial analysis with DSPy."""
    print("=" * 70)
    print("FINANCIAL ANALYSIS WITH YAHOO FINANCE AND DSPY")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configure DSPy
    print_step("Setting up Language Model", "Configuring DSPy for financial analysis")
    try:
        lm = setup_default_lm(provider="openai", model="gpt-4o-mini", max_tokens=2000)
        dspy.configure(lm=lm)
        print_result("Language model configured successfully!")
    except Exception as e:
        print_error(f"Failed to configure language model: {e}")
        return
    
    # Data Fetcher Class
    class FinancialDataFetcher:
        """Fetches and processes financial data from Yahoo Finance."""
        
        def __init__(self):
            self.cache = {}
        
        def get_stock_info(self, symbol: str) -> Dict[str, Any]:
            """Get basic stock information."""
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0)
                }
            except Exception as e:
                print_error(f"Error fetching data for {symbol}: {e}")
                return {}
        
        def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
            """Get historical price data."""
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                return data
            except Exception as e:
                print_error(f"Error fetching historical data for {symbol}: {e}")
                return pd.DataFrame()
        
        def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
            """Calculate basic technical indicators."""
            if data.empty:
                return {}
            
            # Simple moving averages
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # Volatility (standard deviation of returns)
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # RSI calculation (simplified)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'volatility': volatility,
                'rsi': rsi,
                'current_price': data['Close'].iloc[-1],
                'price_change_1d': (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
            }
    
    # DSPy Signatures
    class StockAnalysis(dspy.Signature):
        """Analyze a stock based on financial data and provide investment insights."""
        
        stock_info = dspy.InputField(desc="Basic stock information including name, sector, financial metrics")
        technical_indicators = dspy.InputField(desc="Technical indicators like RSI, moving averages, volatility")
        market_context = dspy.InputField(desc="Current market conditions and relevant news")
        
        analysis = dspy.OutputField(desc="Comprehensive stock analysis covering strengths and weaknesses")
        recommendation = dspy.OutputField(desc="Investment recommendation: BUY, HOLD, or SELL with reasoning")
        risk_level = dspy.OutputField(desc="Risk assessment: LOW, MEDIUM, or HIGH")
        target_price = dspy.OutputField(desc="Estimated target price for the next 12 months")
    
    class PortfolioOptimization(dspy.Signature):
        """Optimize a portfolio allocation based on stocks and investor profile."""
        
        stocks_data = dspy.InputField(desc="List of stocks with their financial metrics and analysis")
        investor_profile = dspy.InputField(desc="Investor risk tolerance, investment horizon, and goals")
        portfolio_size = dspy.InputField(desc="Total portfolio value for allocation")
        
        allocation = dspy.OutputField(desc="Recommended percentage allocation for each stock")
        rationale = dspy.OutputField(desc="Explanation for the allocation strategy")
        expected_return = dspy.OutputField(desc="Expected annual return percentage")
        risk_metrics = dspy.OutputField(desc="Portfolio risk assessment and diversification analysis")
    
    # Financial Analyzer Module
    class FinancialAnalyzer(dspy.Module):
        """Comprehensive financial analysis module using DSPy and Yahoo Finance."""
        
        def __init__(self):
            super().__init__()
            self.data_fetcher = FinancialDataFetcher()
            self.stock_analyzer = dspy.ChainOfThought(StockAnalysis)
            self.portfolio_optimizer = dspy.ChainOfThought(PortfolioOptimization)
        
        def analyze_stock(self, symbol: str, market_context: str = "Normal market conditions") -> dspy.Prediction:
            """Analyze a single stock."""
            # Fetch stock data
            stock_info = self.data_fetcher.get_stock_info(symbol)
            historical_data = self.data_fetcher.get_historical_data(symbol, "6mo")
            technical_indicators = self.data_fetcher.calculate_technical_indicators(historical_data)
            
            # Format data for analysis
            stock_info_str = f"""
            Company: {stock_info.get('name', 'N/A')} ({symbol})
            Sector: {stock_info.get('sector', 'N/A')}
            Industry: {stock_info.get('industry', 'N/A')}
            Market Cap: ${stock_info.get('market_cap', 0):,}
            Current Price: ${stock_info.get('current_price', 0):.2f}
            P/E Ratio: {stock_info.get('pe_ratio', 0):.2f}
            Dividend Yield: {stock_info.get('dividend_yield', 0):.2%}
            Beta: {stock_info.get('beta', 0):.2f}
            52-Week Range: ${stock_info.get('52_week_low', 0):.2f} - ${stock_info.get('52_week_high', 0):.2f}
            """
            
            technical_str = f"""
            Current Price: ${technical_indicators.get('current_price', 0):.2f}
            20-Day SMA: ${technical_indicators.get('sma_20', 0):.2f}
            50-Day SMA: ${technical_indicators.get('sma_50', 0):.2f}
            RSI: {technical_indicators.get('rsi', 0):.2f}
            Volatility (Annualized): {technical_indicators.get('volatility', 0):.2%}
            1-Day Price Change: {technical_indicators.get('price_change_1d', 0):.2f}%
            """
            
            # Perform analysis
            analysis = self.stock_analyzer(
                stock_info=stock_info_str,
                technical_indicators=technical_str,
                market_context=market_context
            )
            
            return analysis
        
        def optimize_portfolio(self, symbols: List[str], investor_profile: str, portfolio_value: float) -> dspy.Prediction:
            """Optimize portfolio allocation for given stocks."""
            stocks_data = []
            
            for symbol in symbols:
                try:
                    analysis = self.analyze_stock(symbol)
                    stock_summary = f"""
                    {symbol}: {analysis.recommendation}
                    Risk Level: {analysis.risk_level}
                    Target Price: {analysis.target_price}
                    Analysis: {analysis.analysis[:200]}...
                    """
                    stocks_data.append(stock_summary)
                except Exception as e:
                    print_error(f"Error analyzing {symbol}: {e}")
            
            stocks_data_str = "\\n\\n".join(stocks_data)
            
            optimization = self.portfolio_optimizer(
                stocks_data=stocks_data_str,
                investor_profile=investor_profile,
                portfolio_size=f"${portfolio_value:,.2f}"
            )
            
            return optimization
    
    # Initialize the financial analyzer
    analyzer = FinancialAnalyzer()
    print_result("Financial analyzer initialized successfully!")
    
    # Stock Analysis Demo
    print_step("Stock Analysis Demo", "Analyzing popular stocks")
    
    stocks_to_analyze = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for symbol in stocks_to_analyze:
        try:
            print(f"\\n{'='*60}")
            print(f"Analyzing {symbol}")
            print('='*60)
            
            analysis = analyzer.analyze_stock(symbol, "Current market showing mixed signals with tech sector volatility")
            
            print_result(f"Analysis: {analysis.analysis}", f"{symbol} Analysis")
            print_result(f"Recommendation: {analysis.recommendation}", "Investment Recommendation")
            print_result(f"Risk Level: {analysis.risk_level}", "Risk Assessment")
            print_result(f"Target Price: {analysis.target_price}", "Price Target")
            
        except Exception as e:
            print_error(f"Error analyzing {symbol}: {e}")
            continue
    
    # Portfolio Optimization Demo
    print_step("Portfolio Optimization Demo", "Creating optimized portfolio allocation")
    
    # Define investor profiles
    investor_profiles = {
        "conservative": """
        Conservative investor, age 55, nearing retirement.
        Risk tolerance: LOW
        Investment horizon: 5-10 years
        Goals: Capital preservation with modest growth
        Prefers: Dividend-paying stocks, low volatility
        """,
        
        "moderate": """
        Moderate investor, age 35, mid-career professional.
        Risk tolerance: MEDIUM
        Investment horizon: 15-20 years
        Goals: Balanced growth and income
        Prefers: Mix of growth and value stocks
        """,
        
        "aggressive": """
        Aggressive investor, age 25, early career.
        Risk tolerance: HIGH
        Investment horizon: 30+ years
        Goals: Maximum long-term growth
        Prefers: Growth stocks, emerging technologies
        """
    }
    
    # Portfolio stocks
    portfolio_stocks = ['AAPL', 'GOOGL', 'MSFT', 'JNJ', 'JPM']
    portfolio_value = 100000  # $100,000 portfolio
    
    for profile_name, profile_desc in investor_profiles.items():
        try:
            print(f"\\n{'='*60}")
            print(f"Portfolio Optimization - {profile_name.title()} Investor")
            print('='*60)
            
            optimization = analyzer.optimize_portfolio(
                symbols=portfolio_stocks,
                investor_profile=profile_desc,
                portfolio_value=portfolio_value
            )
            
            print_result(f"Allocation: {optimization.allocation}", "Recommended Allocation")
            print_result(f"Rationale: {optimization.rationale}", "Strategy Rationale")
            print_result(f"Expected Return: {optimization.expected_return}", "Expected Performance")
            print_result(f"Risk Assessment: {optimization.risk_metrics}", "Risk Analysis")
            
        except Exception as e:
            print_error(f"Error optimizing portfolio for {profile_name}: {e}")
            continue
    
    # Performance Analysis
    print_step("Performance Analysis", "Calculating risk and return metrics")
    
    data_fetcher = FinancialDataFetcher()
    
    for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
        try:
            data = data_fetcher.get_historical_data(symbol, "1y")
            if not data.empty:
                # Calculate annual return and volatility
                price_returns = data['Close'].pct_change().dropna()
                annual_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                annual_volatility = price_returns.std() * np.sqrt(252) * 100
                
                print_result(f"{symbol}: Return: {annual_return:.2f}%, Volatility: {annual_volatility:.2f}%", "Performance Metrics")
                
        except Exception as e:
            print_error(f"Error calculating metrics for {symbol}: {e}")
    
    print("\\n" + "="*70)
    print("FINANCIAL ANALYSIS COMPLETE")
    print("="*70)
    print_result("Successfully demonstrated DSPy-powered financial analysis!")

if __name__ == "__main__":
    main()
