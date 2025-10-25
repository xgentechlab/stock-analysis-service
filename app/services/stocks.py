"""
Stock data fetching and fundamental analysis service
Uses yfinance as the primary data source and database for universe management
"""
import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import time
import random

from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)

class StocksService:
    def __init__(self):
        # NSE suffix for yfinance
        self.nse_suffix = ".NS"
        
        # Rate limiting for yfinance
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Cache for universe data to avoid repeated DB calls
        self._universe_cache = {}
        self._cache_timestamp = None
        self._cache_duration = 300  # 5 minutes cache
        
        # Fallback universe for emergency cases (minimal set)
        self._fallback_universe = [
            "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
            "INFOSYS", "SBIN", "LICI", "ITC", "HINDUNILVR",
            "LT", "HCLTECH", "MARUTI", "SUNPHARMA", "TITAN",
            "ONGC", "TATAMOTORS", "NTPC", "AXISBANK", "NESTLEIND"
        ]
    
    def _get_universe_from_db(self, market_cap_tier: str = "all") -> List[str]:
        """Fetch universe symbols from database with caching"""
        try:
            # Check cache first
            current_time = time.time()
            cache_key = f"universe_{market_cap_tier}"
            
            if (self._cache_timestamp and 
                current_time - self._cache_timestamp < self._cache_duration and 
                cache_key in self._universe_cache):
                logger.debug(f"Using cached universe for {market_cap_tier}")
                return self._universe_cache[cache_key]
            
            # Fetch from database
            logger.info(f"Fetching universe from database for tier: {market_cap_tier}")
            
            if market_cap_tier == "all":
                result = firestore_client.list_stocks(is_active=True, limit=10000)
                stocks = result.get("stocks", [])
            else:
                # For specific tiers, we'll use all stocks and let the calling method filter
                result = firestore_client.list_stocks(is_active=True, limit=10000)
                stocks = result.get("stocks", [])
            
            # Extract symbols
            symbols = [stock.get("symbol") for stock in stocks if stock.get("symbol")]
            
            # Update cache
            self._universe_cache[cache_key] = symbols
            self._cache_timestamp = current_time
            
            logger.info(f"Fetched {len(symbols)} symbols from database for {market_cap_tier}")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching universe from database: {e}")
            # Fallback to minimal emergency list
            logger.warning("Falling back to emergency universe")
            return self._fallback_universe
    
    def get_universe_symbols(self, limit: int = 200) -> List[str]:
        """Get the stock universe for analysis (backward compatibility)"""
        try:
            # Try to get from database first
            symbols = self._get_universe_from_db("all")
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Error getting universe symbols: {e}")
            # Fallback to emergency list
            return self._fallback_universe[:limit]
    
    def get_expanded_universe_symbols(self, limit: int = 500, market_cap_tier: str = "all") -> List[str]:
        """Get expanded universe with market cap filtering from database"""
        try:
            # Get all symbols from database
            all_symbols = self._get_universe_from_db("all")
            
            if market_cap_tier == "all":
                return all_symbols[:limit]
            else:
                # For now, we don't have market cap data in our stocks table
                # So we'll return all symbols and let the calling code handle filtering
                # In the future, we can add market_cap field to stocks and filter here
                logger.warning(f"Market cap tier filtering not implemented yet, returning all symbols")
                return all_symbols[:limit]
                
        except Exception as e:
            logger.error(f"Error getting expanded universe symbols: {e}")
            # Fallback to emergency list
            return self._fallback_universe[:limit]
    
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
            # Get trending stocks from database
            try:
                symbols = self._get_universe_from_db("all")
                trending = symbols[:limit]
            except Exception as e:
                logger.error(f"Error fetching trending stocks from DB: {e}")
                # Fallback to emergency list
                trending = self._fallback_universe[:limit]
            
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
        """Get enhanced stock information with optimized single-pass data fetching"""
        try:
            from app.services.enhanced_indicators import enhanced_technical
            from app.services.enhanced_fundamentals import enhanced_fundamental_analysis
            from app.services.fundamental_scoring import fundamental_scoring
            from app.db.firestore_client import firestore_client
            from datetime import datetime, timezone
            
            # Single data fetch - get all data in one pass
            ticker_data = self._fetch_all_data_optimized(symbol)
            if not ticker_data:
                return {}
            
            # Process technical analysis from fetched data (now includes scores)
            technical_analysis_result = self._process_technical_analysis_optimized(ticker_data, symbol)
            technical_indicators = technical_analysis_result.get("technical_indicators", {})
            technical_scores = technical_analysis_result.get("technical_scores", {})
            
            # Process fundamental analysis from fetched data
            enhanced_fundamentals = self._process_fundamental_analysis_optimized(ticker_data, symbol)
            
            # Calculate fundamental score
            fundamental_score_data = {}
            if enhanced_fundamentals:
                fundamental_score_data = fundamental_scoring.calculate_fundamental_score(enhanced_fundamentals)
            
            # Store multi-timeframe analysis in database
            # Create proper multi-timeframe analysis structure
            mtf_analysis_data = {
                "technical_indicators": technical_indicators,
                "trend_alignment": technical_analysis_result.get("trend_alignment", {}),
                "momentum_scores": technical_analysis_result.get("momentum_scores", {}),
                "divergence_signals": technical_analysis_result.get("divergence_signals", {}),
                "volume_analysis": technical_analysis_result.get("volume_analysis", {}),
                "mtf_score": technical_analysis_result.get("mtf_score"),
                "mtf_confidence": technical_analysis_result.get("mtf_confidence"),
                "mtf_strength": technical_analysis_result.get("mtf_strength")
            }
            mtf_analysis_id = self._store_multi_timeframe_analysis(symbol, ticker_data, mtf_analysis_data)
            
            # Combine all analysis
            enhanced_info = {
                "symbol": symbol,
                "ohlcv": ticker_data["ohlcv_60d"],
                "fundamentals": ticker_data["basic_fundamentals"],
                "current_price": ticker_data["current_price"],
                "data_points": len(ticker_data["ohlcv_60d"]) if ticker_data["ohlcv_60d"] is not None else 0,
                "enhanced_technical": technical_indicators,
                "enhanced_technical_scores": technical_scores,
                "enhanced_fundamentals": enhanced_fundamentals,
                "fundamental_score": fundamental_score_data,
                "multi_timeframe_analysis_id": mtf_analysis_id,
                "data_fetch_optimized": True
            }
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Error getting enhanced stock info for {symbol}: {e}")
            # Fallback to basic data if enhanced fails
            return self._get_basic_fallback(symbol)
    
    def _fetch_all_data_optimized(self, symbol: str) -> Dict[str, Any]:
        """Fetch all required data in a single optimized pass"""
        try:
            # Handle symbol format
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                delay = self.min_request_interval - time_since_last_request + random.uniform(0.2, 0.5)
                time.sleep(delay)
            
            self.last_request_time = time.time()
            
            # Fetch all data in parallel where possible
            logger.info(f"Fetching all data for {symbol} in optimized pass")
            
            # Basic data
            info = ticker.info
            hist_60d = ticker.history(period="2mo")  # 60 days
            hist_1d = ticker.history(period="1d")    # Current price
            
            # Multi-timeframe data
            hist_1m = ticker.history(period="7d", interval="1m")
            hist_5m = ticker.history(period="7d", interval="5m")
            hist_15m = ticker.history(period="7d", interval="15m")
            hist_1wk = ticker.history(period="1y", interval="1wk")
            
            # Financial statements for enhanced fundamentals
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Process OHLCV data
            ohlcv_60d = None
            if not hist_60d.empty and len(hist_60d) >= 30:
                ohlcv_60d = hist_60d[['Open', 'High', 'Low', 'Close', 'Volume']].tail(60)
            
            # Get current price
            current_price = None
            if not hist_1d.empty:
                current_price = float(hist_1d['Close'].iloc[-1])
            elif ohlcv_60d is not None:
                current_price = float(ohlcv_60d['Close'].iloc[-1])
            
            # Basic fundamentals
            basic_fundamentals = {}
            if info:
                basic_fundamentals = {
                    "pe": self._safe_get_float(info, "forwardPE") or self._safe_get_float(info, "trailingPE"),
                    "pb": self._safe_get_float(info, "priceToBook"),
                    "roe": self._safe_get_float(info, "returnOnEquity"),
                    "eps_ttm": self._safe_get_float(info, "trailingEps"),
                    "market_cap_cr": self._safe_get_float(info, "marketCap", convert_to_cr=True)
                }
            
            return {
                "ohlcv_60d": ohlcv_60d,
                "ohlcv_1m": hist_1m if not hist_1m.empty else None,
                "ohlcv_5m": hist_5m if not hist_5m.empty else None,
                "ohlcv_15m": hist_15m if not hist_15m.empty else None,
                "ohlcv_1d": hist_60d if not hist_60d.empty else None,
                "ohlcv_1wk": hist_1wk if not hist_1wk.empty else None,
                "current_price": current_price,
                "basic_fundamentals": basic_fundamentals,
                "info": info,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            }
            
        except Exception as e:
            logger.error(f"Error in optimized data fetch for {symbol}: {e}")
            return {}
    
    def _process_technical_analysis_optimized(self, ticker_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process technical analysis from pre-fetched data"""
        try:
            from app.services.enhanced_indicators import enhanced_technical
            from app.services.enhanced_scoring import enhanced_scoring
            
            # Use the existing enhanced technical analysis but with pre-fetched data
            # We'll modify the enhanced_indicators to accept pre-fetched data
            ohlcv_60d = ticker_data.get("ohlcv_60d")
            if ohlcv_60d is None or ohlcv_60d.empty:
                return {}
            
            # Create multi-timeframe data structure for enhanced_indicators
            mtf_data = {
                '1m': ticker_data.get("ohlcv_1m"),
                '5m': ticker_data.get("ohlcv_5m"),
                '15m': ticker_data.get("ohlcv_15m"),
                '1d': ticker_data.get("ohlcv_1d"),
                '1wk': ticker_data.get("ohlcv_1wk")
            }
            
            # Process technical analysis with pre-fetched data
            technical_analysis = enhanced_technical._analyze_with_prefetched_data(symbol, ohlcv_60d, mtf_data)
            
            # Calculate technical scores (same as in hot stocks endpoint)
            technical_scores = {}
            if technical_analysis:
                try:
                    technical_scores = enhanced_scoring.calculate_enhanced_score(technical_analysis)
                except Exception as e:
                    logger.warning(f"Error calculating technical scores for {symbol}: {e}")
                    technical_scores = {}
            
            # Calculate additional multi-timeframe analysis components
            mtf_trend_analysis = self._calculate_mtf_trend_analysis(mtf_data)
            mtf_momentum_scores = self._calculate_mtf_momentum_scores(mtf_data)
            divergence_analysis = self._calculate_divergence_analysis(technical_analysis, mtf_data)
            volume_analysis = self._calculate_mtf_volume_analysis(mtf_data)
            overall_mtf_score = self._calculate_overall_mtf_score(technical_analysis, mtf_trend_analysis, mtf_momentum_scores)
            mtf_confidence = self._calculate_mtf_confidence(technical_analysis, mtf_trend_analysis)
            mtf_strength = self._calculate_mtf_strength(overall_mtf_score, mtf_confidence)
            
            # Return complete technical analysis with MTF data
            return {
                "technical_indicators": technical_analysis,
                "technical_scores": technical_scores,
                "trend_alignment": mtf_trend_analysis,
                "momentum_scores": mtf_momentum_scores,
                "divergence_signals": divergence_analysis,
                "volume_analysis": volume_analysis,
                "mtf_score": overall_mtf_score,
                "mtf_confidence": mtf_confidence,
                "mtf_strength": mtf_strength
            }
            
        except Exception as e:
            logger.error(f"Error processing technical analysis for {symbol}: {e}")
            return {}
    
    def _process_fundamental_analysis_optimized(self, ticker_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process fundamental analysis from pre-fetched data"""
        try:
            from app.services.enhanced_fundamentals import enhanced_fundamental_analysis
            
            # Use pre-fetched data for enhanced fundamentals
            info = ticker_data.get("info", {})
            financials = ticker_data.get("financials")
            balance_sheet = ticker_data.get("balance_sheet")
            cash_flow = ticker_data.get("cash_flow")
            
            if not info:
                return {}
            
            # Process enhanced fundamentals with pre-fetched data
            enhanced_fundamentals = enhanced_fundamental_analysis._process_with_prefetched_data(
                symbol, info, financials, balance_sheet, cash_flow
            )
            
            return enhanced_fundamentals
            
        except Exception as e:
            logger.error(f"Error processing fundamental analysis for {symbol}: {e}")
            return {}
    
    def _calculate_mtf_trend_analysis(self, mtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend analysis across multiple timeframes"""
        try:
            trend_analysis = {}
            
            for tf_name, data in mtf_data.items():
                if data is None or data.empty or len(data) < 20:
                    continue
                
                close = data['Close']
                sma_20 = close.rolling(20).mean()
                
                # Determine trend direction
                if close.iloc[-1] > sma_20.iloc[-1]:
                    trend = "bullish"
                elif close.iloc[-1] < sma_20.iloc[-1]:
                    trend = "bearish"
                else:
                    trend = "neutral"
                
                # Calculate trend strength
                trend_strength = abs(close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100
                
                trend_analysis[tf_name] = {
                    "trend": trend,
                    "strength": trend_strength,
                    "price": float(close.iloc[-1]),
                    "sma_20": float(sma_20.iloc[-1])
                }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error calculating MTF trend analysis: {e}")
            return {}
    
    def _calculate_mtf_momentum_scores(self, mtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate momentum scores for each timeframe"""
        try:
            momentum_scores = {}
            
            for tf_name, data in mtf_data.items():
                if data is None or data.empty or len(data) < 10:
                    continue
                
                close = data['Close']
                
                # Calculate various momentum metrics
                if len(close) >= 5:
                    momentum_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
                else:
                    momentum_5 = 0
                
                if len(close) >= 10:
                    momentum_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100
                else:
                    momentum_10 = 0
                
                if len(close) >= 20:
                    momentum_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
                else:
                    momentum_20 = 0
                
                # Calculate RSI for momentum
                if len(close) >= 14:
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi_value = float(rsi.iloc[-1]) if not rsi.empty else 50
                else:
                    rsi_value = 50
                
                momentum_scores[tf_name] = {
                    "momentum_5": momentum_5,
                    "momentum_10": momentum_10,
                    "momentum_20": momentum_20,
                    "rsi": rsi_value,
                    "overall_score": (momentum_5 + momentum_10 + momentum_20) / 3
                }
            
            return momentum_scores
            
        except Exception as e:
            logger.error(f"Error calculating MTF momentum scores: {e}")
            return {}
    
    def _calculate_divergence_analysis(self, technical_analysis: Dict[str, Any], mtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate divergence analysis across timeframes"""
        try:
            divergence_signals = {
                "bullish_divergence": False,
                "bearish_divergence": False,
                "timeframe_divergences": {}
            }
            
            # Check for RSI divergence
            rsi_values = []
            price_values = []
            
            for tf_name, data in mtf_data.items():
                if data is None or data.empty or len(data) < 20:
                    continue
                
                close = data['Close']
                if len(close) >= 14:
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    if not rsi.empty:
                        rsi_values.append(float(rsi.iloc[-1]))
                        price_values.append(float(close.iloc[-1]))
            
            # Simple divergence detection
            if len(rsi_values) >= 2 and len(price_values) >= 2:
                price_trend = price_values[-1] > price_values[0]
                rsi_trend = rsi_values[-1] > rsi_values[0]
                
                if price_trend and not rsi_trend:
                    divergence_signals["bearish_divergence"] = True
                elif not price_trend and rsi_trend:
                    divergence_signals["bullish_divergence"] = True
            
            return divergence_signals
            
        except Exception as e:
            logger.error(f"Error calculating divergence analysis: {e}")
            return {"bullish_divergence": False, "bearish_divergence": False, "timeframe_divergences": {}}
    
    def _calculate_mtf_volume_analysis(self, mtf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate volume analysis across timeframes"""
        try:
            volume_analysis = {}
            
            for tf_name, data in mtf_data.items():
                if data is None or data.empty or len(data) < 10:
                    continue
                
                volume = data['Volume']
                
                # Calculate volume metrics
                avg_volume = volume.mean()
                current_volume = volume.iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Volume trend
                if len(volume) >= 5:
                    volume_trend = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5] * 100
                else:
                    volume_trend = 0
                
                volume_analysis[tf_name] = {
                    "current_volume": float(current_volume),
                    "avg_volume": float(avg_volume),
                    "volume_ratio": float(volume_ratio),
                    "volume_trend": float(volume_trend),
                    "volume_score": min(1.0, volume_ratio)  # Normalize to 0-1
                }
            
            return volume_analysis
            
        except Exception as e:
            logger.error(f"Error calculating MTF volume analysis: {e}")
            return {}
    
    def _calculate_overall_mtf_score(self, technical_analysis: Dict[str, Any], trend_analysis: Dict[str, Any], momentum_scores: Dict[str, Any]) -> float:
        """Calculate overall multi-timeframe score"""
        try:
            scores = []
            
            # Add trend scores
            for tf_data in trend_analysis.values():
                if tf_data.get("trend") == "bullish":
                    scores.append(0.7 + min(0.3, tf_data.get("strength", 0) / 10))
                elif tf_data.get("trend") == "bearish":
                    scores.append(0.3 - min(0.3, tf_data.get("strength", 0) / 10))
                else:
                    scores.append(0.5)
            
            # Add momentum scores
            for tf_data in momentum_scores.values():
                overall_score = tf_data.get("overall_score", 0)
                normalized_score = 0.5 + (overall_score / 100) * 0.5  # Normalize to 0-1
                scores.append(normalized_score)
            
            # Add technical analysis score
            if technical_analysis:
                rsi = technical_analysis.get("rsi_14", 50)
                rsi_score = 1 - abs(rsi - 50) / 50  # RSI closer to 50 = better score
                scores.append(rsi_score)
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating overall MTF score: {e}")
            return 0.5
    
    def _calculate_mtf_confidence(self, technical_analysis: Dict[str, Any], trend_analysis: Dict[str, Any]) -> float:
        """Calculate confidence level for MTF analysis"""
        try:
            confidence_factors = []
            
            # Data availability factor
            data_quality = len(trend_analysis) / 5.0  # 5 timeframes expected
            confidence_factors.append(data_quality)
            
            # Technical indicator consistency
            if technical_analysis:
                rsi = technical_analysis.get("rsi_14", 50)
                macd = technical_analysis.get("macd", 0)
                macd_signal = technical_analysis.get("macd_signal", 0)
                
                # RSI confidence (closer to extremes = higher confidence)
                rsi_confidence = abs(rsi - 50) / 50
                confidence_factors.append(rsi_confidence)
                
                # MACD confidence
                macd_confidence = abs(macd - macd_signal) / max(abs(macd), abs(macd_signal), 1)
                confidence_factors.append(min(1.0, macd_confidence))
            
            # Trend alignment factor
            if trend_analysis:
                bullish_count = sum(1 for tf in trend_analysis.values() if tf.get("trend") == "bullish")
                bearish_count = sum(1 for tf in trend_analysis.values() if tf.get("trend") == "bearish")
                total_trends = bullish_count + bearish_count
                
                if total_trends > 0:
                    trend_alignment = max(bullish_count, bearish_count) / total_trends
                    confidence_factors.append(trend_alignment)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating MTF confidence: {e}")
            return 0.5
    
    def _calculate_mtf_strength(self, mtf_score: float, mtf_confidence: float) -> str:
        """Calculate MTF strength rating"""
        try:
            combined_score = (mtf_score + mtf_confidence) / 2
            
            if combined_score >= 0.8:
                return "very_strong"
            elif combined_score >= 0.6:
                return "strong"
            elif combined_score >= 0.4:
                return "moderate"
            elif combined_score >= 0.2:
                return "weak"
            else:
                return "very_weak"
                
        except Exception as e:
            logger.error(f"Error calculating MTF strength: {e}")
            return "unknown"
    
    def _store_multi_timeframe_analysis(self, symbol: str, ticker_data: Dict[str, Any], technical_analysis: Dict[str, Any]) -> str:
        """Store multi-timeframe analysis in database"""
        try:
            from app.db.firestore_client import firestore_client
            from datetime import datetime, timezone
            import uuid
            
            analysis_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            # Create multi-timeframe analysis document
            mtf_analysis = {
                "symbol": symbol,
                "analysis_id": analysis_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "technical_indicators": technical_analysis.get("technical_indicators", {}),
                "trend_analysis": technical_analysis.get("trend_alignment", {}),  # Fixed field name
                "momentum_scores": technical_analysis.get("momentum_scores", {}),
                "divergence_signals": technical_analysis.get("divergence_signals", {}),
                "volume_analysis": technical_analysis.get("volume_analysis", {}),
                "mtf_scores": {  # Fixed to mtf_scores (plural)
                    "overall_score": technical_analysis.get("mtf_score"),
                    "confidence": technical_analysis.get("mtf_confidence"),
                    "strength": technical_analysis.get("mtf_strength")
                },
                "analysis_version": "1.0",
                "data_quality": "good"
            }
            
            # Store in database
            firestore_client.create_multi_timeframe_analysis(mtf_analysis)
            
            logger.info(f"Stored multi-timeframe analysis {analysis_id} for {symbol}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error storing multi-timeframe analysis for {symbol}: {e}")
            return ""
    
    def _get_basic_fallback(self, symbol: str) -> Dict[str, Any]:
        """Fallback to basic data if enhanced analysis fails"""
        try:
            # Simple fallback - just get basic data
            ohlcv = self.fetch_ohlcv_data(symbol, days=60)
            fundamentals = self.fetch_fundamentals(symbol)
            current_price = self.get_current_price(symbol)
            
            return {
                "symbol": symbol,
                "ohlcv": ohlcv,
                "fundamentals": fundamentals,
                "current_price": current_price,
                "data_points": len(ohlcv) if ohlcv is not None else 0,
                "enhanced_technical": {},
                "enhanced_fundamentals": {},
                "fundamental_score": {},
                "multi_timeframe_analysis_id": "",
                "data_fetch_optimized": False
            }
            
        except Exception as e:
            logger.error(f"Error in basic fallback for {symbol}: {e}")
            return {}

# Singleton instance
stocks_service = StocksService()
