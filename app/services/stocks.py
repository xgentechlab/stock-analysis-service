"""
Stock data fetching and fundamental analysis service
Uses yfinance as the primary data source
"""
import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import time
import random

logger = logging.getLogger(__name__)

class StocksService:
    def __init__(self):
        # NSE suffix for yfinance
        self.nse_suffix = ".NS"
        
        # Rate limiting for yfinance
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Top 200 NSE stocks by market cap (sample list - should be updated monthly)
        # In production, this should be fetched from NSE API or screener
        self.top_200_universe = [
            "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
            "INFOSYS", "SBIN", "LICI", "ITC", "HINDUNILVR",
            "LT", "HCLTECH", "MARUTI", "SUNPHARMA", "TITAN",
            "ONGC", "TATAMOTORS", "NTPC", "AXISBANK", "NESTLEIND",
            "WIPRO", "ULTRACEMCO", "ADANIENT", "ASIANPAINT", "BAJFINANCE",
            "M&M", "TATACONSUM", "BAJAJFINSV", "POWERGRID", "TECHM",
            "COALINDIA", "INDUSINDBK", "DRREDDY", "GRASIM", "TATASTEEL",
            "CIPLA", "JSWSTEEL", "BRITANNIA", "EICHERMOT", "HEROMOTOCO",
            "APOLLOHOSP", "DIVISLAB", "BPCL", "GODREJCP", "PIDILITIND",
            "DABUR", "MARICO", "BERGEPAINT", "COLPAL", "MCDOWELL"
            # ... Add more stocks to reach 200
        ]
    
    def get_universe_symbols(self, limit: int = 200) -> List[str]:
        """Get the stock universe for analysis"""
        return self.top_200_universe[:limit]
    
    def fetch_ohlcv_data(self, symbol: str, days: int = 60, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol using yfinance
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        """
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '')
            
            logger.info(f"Fetching OHLCV data for {clean_symbol} using yfinance")
            
            # Handle both formats: "RELIANCE" -> "RELIANCE.NS" or "RELIANCE.NS" -> "RELIANCE.NS"
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            
            # Test network connectivity first
            logger.info(f"Testing network connectivity for {ticker_symbol}")
            try:
                response = requests.get("https://www.google.com", timeout=10)
                logger.info(f"Network test successful: {response.status_code}")
            except Exception as e:
                logger.error(f"Network connectivity test failed: {e}")
                raise ValueError(f"Network connectivity issue: {e}")
            
            # Rate limiting for yfinance
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                delay = self.min_request_interval - time_since_last_request + random.uniform(0.2, 0.5)
                logger.info(f"Rate limiting: waiting {delay:.2f}s since last request")
                time.sleep(delay)
            
            self.last_request_time = time.time()
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends
            
            # Fetch data with retry logic
            hist = None
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date} (attempt {attempt + 1}/{max_retries})")
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    logger.info(f"Raw data shape: {hist.shape}, columns: {list(hist.columns)}")
                    
                    if not hist.empty:
                        break  # Success, exit retry loop
                    else:
                        logger.warning(f"Empty data on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            retry_delay = (attempt + 1) * 1 + random.uniform(0.5, 1.5)
                            logger.info(f"Retrying in {retry_delay:.2f}s...")
                            time.sleep(retry_delay)
                        
                except Exception as fetch_error:
                    logger.error(f"Fetch attempt {attempt + 1} failed: {fetch_error}")
                    if attempt < max_retries - 1:
                        retry_delay = (attempt + 1) * 1.5 + random.uniform(1, 2)
                        logger.info(f"Retrying in {retry_delay:.2f}s...")
                        time.sleep(retry_delay)
                    else:
                        raise fetch_error
            
            if hist is None or hist.empty:
                logger.warning(f"No OHLCV data found for {symbol} (ticker: {ticker_symbol}) after {max_retries} attempts")
                # Try to get more info about the ticker
                try:
                    info = ticker.info
                    logger.info(f"Ticker info available: {bool(info)}")
                    if info:
                        logger.info(f"Ticker name: {info.get('longName', 'N/A')}")
                        logger.info(f"Ticker symbol: {info.get('symbol', 'N/A')}")
                except Exception as info_error:
                    logger.error(f"Could not fetch ticker info: {info_error}")
                return None
            
            # Ensure we have enough data
            if len(hist) < 30:  # Minimum 30 days
                logger.warning(f"Insufficient OHLCV data for {symbol}: {len(hist)} days")
                return None
            
            # Take last 'days' entries
            hist = hist.tail(days)
            
            # Standardize column names - yfinance returns 7 columns, we need 5
            # Original: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
            # We want: ['Open', 'High', 'Low', 'Close', 'Volume']
            hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Fetched {len(hist)} days of OHLCV data for {symbol}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None
    
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental data for a symbol
        In production, this should use a proper financial data API
        For now, using yfinance with fallback values
        """
        try:
            # Handle both formats: "RELIANCE" -> "RELIANCE.NS" or "RELIANCE.NS" -> "RELIANCE.NS"
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            ticker = yf.Ticker(ticker_symbol)
            
            info = ticker.info
            if not info:
                logger.warning(f"No fundamental data found for {symbol}")
                return {}
            
            # Extract fundamental metrics
            fundamentals = {
                "pe": self._safe_get_float(info, "forwardPE") or self._safe_get_float(info, "trailingPE"),
                "pb": self._safe_get_float(info, "priceToBook"),
                "roe": self._safe_get_float(info, "returnOnEquity"),
                "eps_ttm": self._safe_get_float(info, "trailingEps"),
                "market_cap_cr": self._safe_get_float(info, "marketCap", convert_to_cr=True)
            }
            
            logger.info(f"Fetched fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {}
    
    def _safe_get_float(self, data: Dict, key: str, convert_to_cr: bool = False) -> Optional[float]:
        """Safely extract float value from data dict"""
        try:
            value = data.get(key)
            if value is None or value == 'N/A':
                return None
            
            float_value = float(value)
            
            if convert_to_cr:
                # Convert to crores (divide by 10,000,000)
                float_value = float_value / 10000000
            
            return float_value
        except (ValueError, TypeError):
            return None
    
    def check_fundamental_sanity(self, fundamentals: Dict[str, Optional[float]], 
                               min_market_cap_cr: float = 500, 
                               max_pe: float = 60) -> bool:
        """
        Check if fundamental data passes sanity filters
        """
        try:
            market_cap = fundamentals.get("market_cap_cr")
            pe = fundamentals.get("pe")
            eps_ttm = fundamentals.get("eps_ttm")
            
            # Market cap check
            if not market_cap or market_cap < min_market_cap_cr:
                return False
            
            # PE ratio check
            if not pe or pe <= 0 or pe > max_pe:
                return False
            
            # EPS check
            if not eps_ttm or eps_ttm <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in fundamental sanity check: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol using yfinance"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '')
            
            logger.info(f"Fetching current price for {clean_symbol} using yfinance")
            
            # Handle both formats: "RELIANCE" -> "RELIANCE.NS" or "RELIANCE.NS" -> "RELIANCE.NS"
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                delay = self.min_request_interval - time_since_last_request + random.uniform(0.2, 0.5)
                logger.info(f"Rate limiting: waiting {delay:.2f}s since last request")
                time.sleep(delay)
            
            self.last_request_time = time.time()
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Get latest price from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            # Fallback to info
            info = ticker.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if current_price:
                return float(current_price)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def fetch_trending_stocks(self, limit: int = 50) -> List[str]:
        """
        Fetch trending/active stocks
        In production, this could use NSE most active stocks API
        For now, return a subset of universe based on volume/activity
        """
        try:
            # For demo purposes, return first N stocks from universe
            # In production, this should fetch from NSE active stocks API
            trending = self.top_200_universe[:limit]
            
            logger.info(f"Fetched {len(trending)} trending stocks")
            return trending
            
        except Exception as e:
            logger.error(f"Error fetching trending stocks: {e}")
            return []
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            # Fetch OHLCV data
            ohlcv = self.fetch_ohlcv_data(symbol, days=60)
            if ohlcv is None or ohlcv.empty:
                return {}
            
            # Fetch fundamentals
            fundamentals = self.fetch_fundamentals(symbol)
            
            # Get current price
            current_price = self.get_current_price(symbol)
            
            return {
                "symbol": symbol,
                "ohlcv": ohlcv,
                "fundamentals": fundamentals,
                "current_price": current_price,
                "data_points": len(ohlcv) if ohlcv is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            return {}
    
    def get_enhanced_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced stock information with multi-timeframe analysis and enhanced fundamentals"""
        try:
            from app.services.enhanced_indicators import enhanced_technical
            from app.services.enhanced_fundamentals import enhanced_fundamental_analysis
            from app.services.fundamental_scoring import fundamental_scoring
            
            # Get basic stock info
            basic_info = self.get_stock_info(symbol)
            if not basic_info:
                return {}
            
            # Get enhanced technical analysis
            technical_analysis = enhanced_technical.analyze_symbol(symbol, days_back=30)
            
            # Get enhanced fundamental analysis
            enhanced_fundamentals = enhanced_fundamental_analysis.fetch_enhanced_fundamentals(symbol)
            
            # Calculate fundamental score
            fundamental_score_data = {}
            if enhanced_fundamentals:
                fundamental_score_data = fundamental_scoring.calculate_fundamental_score(enhanced_fundamentals)
            
            # Combine basic and enhanced analysis
            enhanced_info = {
                **basic_info,
                "enhanced_technical": technical_analysis,
                "enhanced_fundamentals": enhanced_fundamentals,
                "fundamental_score": fundamental_score_data
            }
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Error getting enhanced stock info for {symbol}: {e}")
            return self.get_stock_info(symbol)  # Fallback to basic info

# Singleton instance
stocks_service = StocksService()
