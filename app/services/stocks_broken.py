"""
Stock data fetching and fundamental analysis service
Configuration-driven data source selection with fallback support
"""
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import time
import random
from .nse_service import NSEService
from .alpha_vantage_service import AlphaVantageService
from .yfinance_service import YFinanceService
from app.config import settings

logger = logging.getLogger(__name__)

class StocksService:
    def __init__(self):
        # Initialize data sources based on configuration
        self.alpha_vantage_service = AlphaVantageService()
        self.nse_service = NSEService()
        self.yfinance_service = YFinanceService()
        
        # Get data source configuration
        self.primary_source = settings.primary_data_source
        self.fallback_source = settings.fallback_data_source
        self.enable_yfinance = settings.enable_yfinance
        
        # Rate limiting for fallback services
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests for fallback services
        
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
        Fetch OHLCV data for a symbol using configuration-driven data source selection
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        """
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '').replace('.BSE', '')
            
            logger.info(f"Fetching OHLCV data for {clean_symbol} using {self.primary_source} as primary source")
            
            # Try primary source first
            primary_data = self._fetch_from_source(self.primary_source, clean_symbol, days, 'ohlcv')
            if primary_data is not None and not primary_data.empty:
                logger.info(f"Successfully fetched data from {self.primary_source} for {clean_symbol}")
                return primary_data
            
            # Try fallback source
            logger.warning(f"{self.primary_source} returned no data for {clean_symbol}, trying {self.fallback_source} fallback")
            fallback_data = self._fetch_from_source(self.fallback_source, clean_symbol, days, 'ohlcv')
            if fallback_data is not None and not fallback_data.empty:
                logger.info(f"Successfully fetched data from {self.fallback_source} for {clean_symbol}")
                return fallback_data
            
            logger.error(f"All data sources failed for {clean_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None
    
    def _fetch_from_source(self, source: str, symbol: str, days: int, data_type: str) -> Optional[pd.DataFrame]:
        """Fetch data from specified source"""
        try:
            if source == "alpha_vantage":
                if data_type == 'ohlcv':
                    return self.alpha_vantage_service.fetch_ohlcv_data(symbol, days)
                elif data_type == 'current_price':
                    price = self.alpha_vantage_service.get_current_price(symbol)
                    return pd.DataFrame({'price': [price]}) if price else None
                elif data_type == 'fundamentals':
                    return self.alpha_vantage_service.get_fundamental_data(symbol)
                elif data_type == 'technical':
                    return self.alpha_vantage_service.get_technical_indicators(symbol)
            
            elif source == "yfinance" and self.enable_yfinance:
                if data_type == 'ohlcv':
                    return self.yfinance_service.fetch_ohlcv_data(symbol, days)
                elif data_type == 'current_price':
                    price = self.yfinance_service.get_current_price(symbol)
                    return pd.DataFrame({'price': [price]}) if price else None
                elif data_type == 'fundamentals':
                    return self.yfinance_service.get_fundamental_data(symbol)
                elif data_type == 'technical':
                    return self.yfinance_service.get_technical_indicators(symbol)
            
            elif source == "nse":
                if data_type == 'ohlcv':
                    return self.nse_service.fetch_ohlcv_data(symbol, days)
                elif data_type == 'current_price':
                    price = self.nse_service.get_current_price(symbol)
                    return pd.DataFrame({'price': [price]}) if price else None
                elif data_type == 'fundamentals':
                    return self.nse_service.get_fundamental_data(symbol)
                elif data_type == 'technical':
                    return self.nse_service.get_technical_indicators(symbol)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching {data_type} from {source} for {symbol}: {e}")
                return None
                if nse_data is not None and not nse_data.empty:
                    logger.info(f"Successfully fetched data from NSE Direct API for {clean_symbol}")
                    return nse_data
                else:
                    logger.warning(f"NSE Direct API returned no data for {clean_symbol}")
            except Exception as e:
                logger.warning(f"NSE Direct API failed for {clean_symbol}: {e}")
            
            logger.error(f"All data sources failed for {clean_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol using Alpha Vantage with NSE fallback"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '').replace('.BSE', '')
            
            # Try Alpha Vantage first
            try:
                av_price = self.alpha_vantage_service.get_current_price(clean_symbol)
                if av_price is not None:
                    logger.info(f"Successfully fetched current price from Alpha Vantage for {clean_symbol}: {av_price}")
                    return av_price
                else:
                    logger.warning(f"Alpha Vantage returned no price for {clean_symbol}, trying NSE fallback")
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {clean_symbol}: {e}, trying NSE fallback")
            
            # Try NSE Direct API as fallback
            try:
                nse_price = self.nse_service.get_current_price(clean_symbol)
                if nse_price is not None:
                    logger.info(f"Successfully fetched current price from NSE Direct API for {clean_symbol}: {nse_price}")
                    return nse_price
                else:
                    logger.warning(f"NSE Direct API returned no price for {clean_symbol}")
            except Exception as e:
                logger.warning(f"NSE Direct API failed for {clean_symbol}: {e}")
            
            logger.error(f"All data sources failed for current price of {clean_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data for a symbol using Alpha Vantage with NSE fallback"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '').replace('.BSE', '')
            
            # Try Alpha Vantage first
            try:
                av_fundamentals = self.alpha_vantage_service.get_fundamental_data(clean_symbol)
                if av_fundamentals is not None and av_fundamentals:
                    logger.info(f"Successfully fetched fundamental data from Alpha Vantage for {clean_symbol}")
                    return av_fundamentals
                else:
                    logger.warning(f"Alpha Vantage returned no fundamental data for {clean_symbol}, trying NSE fallback")
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {clean_symbol}: {e}, trying NSE fallback")
            
            # Try NSE Direct API as fallback (if it has fundamental data)
            try:
                nse_fundamentals = self.nse_service.get_fundamental_data(clean_symbol)
                if nse_fundamentals is not None and nse_fundamentals:
                    logger.info(f"Successfully fetched fundamental data from NSE Direct API for {clean_symbol}")
                    return nse_fundamentals
                else:
                    logger.warning(f"NSE Direct API returned no fundamental data for {clean_symbol}")
            except Exception as e:
                logger.warning(f"NSE Direct API failed for {clean_symbol}: {e}")
            
            logger.error(f"All data sources failed for fundamental data of {clean_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None

    def get_technical_indicators(self, symbol: str, indicators: List[str] = None) -> Optional[Dict[str, Any]]:
        """Get technical indicators for a symbol using Alpha Vantage"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('.NS', '').replace('.BSE', '')
            
            # Use Alpha Vantage for technical indicators
            try:
                av_indicators = self.alpha_vantage_service.get_technical_indicators(clean_symbol, indicators)
                if av_indicators is not None and av_indicators:
                    logger.info(f"Successfully fetched technical indicators from Alpha Vantage for {clean_symbol}")
                    return av_indicators
                else:
                    logger.warning(f"Alpha Vantage returned no technical indicators for {clean_symbol}")
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {clean_symbol}: {e}")
            
            logger.error(f"Failed to fetch technical indicators for {clean_symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching technical indicators for {symbol}: {e}")
            return None

    def check_fundamental_sanity(self, fundamentals: Dict[str, Optional[float]], 
                               min_market_cap_cr: float = 500, 
                               max_pe: float = 60) -> bool:
        """
        Check if fundamental data passes basic sanity checks
        
        Args:
            fundamentals: Dictionary of fundamental metrics
            min_market_cap_cr: Minimum market cap in crores
            max_pe: Maximum PE ratio allowed
        
        Returns:
            True if data passes sanity checks, False otherwise
        """
        try:
            # Check market cap
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap and market_cap < min_market_cap_cr * 10000000:  # Convert crores to actual value
                logger.debug(f"Market cap too low: {market_cap}")
                return False
            
            # Check PE ratio
            pe_ratio = fundamentals.get('pe_ratio', 0)
            if pe_ratio and (pe_ratio <= 0 or pe_ratio > max_pe):
                logger.debug(f"PE ratio out of range: {pe_ratio}")
                return False
            
            # Check EPS
            eps = fundamentals.get('eps', 0)
            if eps and eps <= 0:
                logger.debug(f"EPS is negative or zero: {eps}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in fundamental sanity check: {e}")
            return False

# Singleton instance
stocks_service = StocksService()
