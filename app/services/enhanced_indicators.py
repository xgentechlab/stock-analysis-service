"""
Enhanced Technical Analysis Framework
Multi-timeframe analysis with advanced momentum and volume indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

class MultiTimeframeData:
    """Handles multi-timeframe data fetching and management"""
    
    def __init__(self):
        self.nse_suffix = ".NS"
    
    def fetch_multi_timeframe_data(self, symbol: str, days_back: int = 30) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetch data for multiple timeframes
        Returns dict with timeframe as key and DataFrame as value
        """
        try:
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            
            ticker = yf.Ticker(ticker_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 10)
            
            timeframes = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1d': '1d',
                '1wk': '1wk'
            }
            
            data = {}
            
            for tf_name, tf_interval in timeframes.items():
                try:
                    # For intraday data, limit to recent period
                    if tf_interval in ['1m', '5m', '15m']:
                        period_days = min(7, days_back)  # Limit intraday to 7 days max
                        tf_start = end_date - timedelta(days=period_days)
                    else:
                        tf_start = start_date
                    
                    hist = ticker.history(start=tf_start, end=end_date, interval=tf_interval)
                    
                    if not hist.empty and len(hist) >= 10:  # Minimum data points
                        # Standardize column names
                        hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
                        data[tf_name] = hist
                        logger.debug(f"Fetched {len(hist)} {tf_name} candles for {symbol}")
                    else:
                        logger.debug(f"Insufficient {tf_name} data for {symbol}: {len(hist) if not hist.empty else 0} candles")
                        data[tf_name] = None
                        
                except Exception as e:
                    logger.warning(f"Error fetching {tf_name} data for {symbol}: {e}")
                    data[tf_name] = None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching multi-timeframe data for {symbol}: {e}")
            return {tf: None for tf in ['1m', '5m', '15m', '1d', '1wk']}

class AdvancedMomentumIndicators:
    """Advanced momentum indicators and divergence detection"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with proper handling"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def detect_rsi_divergence(prices: pd.Series, rsi: pd.Series, lookback: int = 20) -> Dict[str, bool]:
        """
        Detect RSI divergence patterns
        Returns dict with divergence signals
        """
        try:
            if len(prices) < lookback * 2:
                return {"bullish_divergence": False, "bearish_divergence": False}
            
            # Get recent data
            recent_prices = prices.tail(lookback)
            recent_rsi = rsi.tail(lookback)
            
            # Find peaks and troughs
            price_peaks = recent_prices[recent_prices == recent_prices.rolling(5, center=True).max()].dropna()
            price_troughs = recent_prices[recent_prices == recent_prices.rolling(5, center=True).min()].dropna()
            
            rsi_peaks = recent_rsi[recent_rsi == recent_rsi.rolling(5, center=True).max()].dropna()
            rsi_troughs = recent_rsi[recent_rsi == recent_rsi.rolling(5, center=True).min()].dropna()
            
            # Check for divergences
            bullish_div = False
            bearish_div = False
            
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                # Bullish divergence: price makes lower low, RSI makes higher low
                last_two_price_troughs = price_troughs.tail(2)
                last_two_rsi_troughs = rsi_troughs.tail(2)
                
                if (last_two_price_troughs.iloc[-1] < last_two_price_troughs.iloc[0] and 
                    last_two_rsi_troughs.iloc[-1] > last_two_rsi_troughs.iloc[0]):
                    bullish_div = True
            
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                # Bearish divergence: price makes higher high, RSI makes lower high
                last_two_price_peaks = price_peaks.tail(2)
                last_two_rsi_peaks = rsi_peaks.tail(2)
                
                if (last_two_price_peaks.iloc[-1] > last_two_price_peaks.iloc[0] and 
                    last_two_rsi_peaks.iloc[-1] < last_two_rsi_peaks.iloc[0]):
                    bearish_div = True
            
            return {
                "bullish_divergence": bullish_div,
                "bearish_divergence": bearish_div
            }
            
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return {"bullish_divergence": False, "bearish_divergence": False}
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD with histogram"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def calculate_stochastic_rsi(prices: pd.Series, rsi_period: int = 14, stoch_period: int = 14, 
                                k_period: int = 3, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic RSI"""
        # First calculate RSI
        rsi = AdvancedMomentumIndicators.calculate_rsi(prices, rsi_period)
        
        # Calculate Stochastic of RSI
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
        stoch_rsi_k = stoch_rsi.rolling(window=k_period).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
        
        return {
            "stoch_rsi": stoch_rsi,
            "stoch_rsi_k": stoch_rsi_k,
            "stoch_rsi_d": stoch_rsi_d
        }
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    @staticmethod
    def calculate_roc(prices: pd.Series, periods: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """Calculate Rate of Change for multiple periods"""
        roc_data = {}
        for period in periods:
            roc_data[f"roc_{period}"] = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc_data

class VolumeProfileAnalysis:
    """Advanced volume analysis and profile indicators"""
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_vwap_bands(vwap: pd.Series, volume: pd.Series, std_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate VWAP standard deviation bands"""
        # Calculate rolling standard deviation of price from VWAP
        price_deviation = (vwap - vwap.shift(1)) ** 2
        weighted_variance = (price_deviation * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        vwap_std = np.sqrt(weighted_variance)
        
        return {
            "vwap": vwap,
            "vwap_upper": vwap + (vwap_std * std_multiplier),
            "vwap_lower": vwap - (vwap_std * std_multiplier)
        }
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0)).cumsum()
        return pd.Series(obv.astype(float), index=close.index)
    
    @staticmethod
    def calculate_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line.astype(float)
    
    @staticmethod
    def calculate_volume_profile(high: pd.Series, low: pd.Series, volume: pd.Series, 
                               price_levels: int = 20) -> Dict[str, Any]:
        """Calculate volume profile for support/resistance identification"""
        try:
            # Create price levels
            price_min = low.min()
            price_max = high.max()
            price_range = price_max - price_min
            level_size = price_range / price_levels
            
            # Initialize volume at each price level
            volume_at_level = np.zeros(price_levels)
            level_prices = []
            
            for i in range(price_levels):
                level_price = price_min + (i * level_size)
                level_prices.append(level_price)
            
            # Distribute volume across price levels
            for idx, (h, l, v) in enumerate(zip(high, low, volume)):
                if pd.isna(h) or pd.isna(l) or pd.isna(v):
                    continue
                    
                # Find which levels this bar touches
                start_level = max(0, int((l - price_min) / level_size))
                end_level = min(price_levels - 1, int((h - price_min) / level_size))
                
                # Distribute volume across touched levels
                if end_level > start_level:
                    volume_per_level = v / (end_level - start_level + 1)
                    for level in range(start_level, end_level + 1):
                        volume_at_level[level] += volume_per_level
            
            # Find high volume nodes (support/resistance)
            mean_volume = np.mean(volume_at_level)
            high_volume_threshold = mean_volume * 1.5
            
            high_volume_levels = []
            for i, vol in enumerate(volume_at_level):
                if vol > high_volume_threshold:
                    high_volume_levels.append({
                        "price": float(level_prices[i]),
                        "volume": float(vol),
                        "level": int(i)
                    })
            
            return {
                "volume_profile": volume_at_level.tolist(),
                "level_prices": level_prices,
                "high_volume_nodes": high_volume_levels,
                "mean_volume": float(mean_volume)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {"volume_profile": [], "level_prices": [], "high_volume_nodes": [], "mean_volume": 0}

class EnhancedTechnicalFramework:
    """Comprehensive technical analysis framework"""
    
    def __init__(self):
        self.mtf_data = MultiTimeframeData()
        self.momentum = AdvancedMomentumIndicators()
        self.volume = VolumeProfileAnalysis()
    
    def analyze_symbol(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Comprehensive technical analysis for a symbol
        Returns enhanced technical analysis results
        """
        try:
            # Fetch multi-timeframe data
            mtf_data = self.mtf_data.fetch_multi_timeframe_data(symbol, days_back)
            
            daily_data = mtf_data.get('1d')
            if daily_data is None or daily_data.empty:
                logger.warning(f"No daily data available for {symbol}")
                return {}
            analysis = {}
            
            # Basic technical indicators
            try:
                analysis.update(self._calculate_basic_indicators(daily_data))
            except Exception as e:
                logger.error(f"Error calculating basic indicators for {symbol}: {e}")
            
            # Advanced momentum analysis
            try:
                analysis.update(self._calculate_momentum_analysis(daily_data))
            except Exception as e:
                logger.error(f"Error calculating momentum analysis for {symbol}: {e}")
            
            # Volume profile analysis
            try:
                analysis.update(self._calculate_volume_analysis(daily_data))
            except Exception as e:
                logger.error(f"Error calculating volume analysis for {symbol}: {e}")
            
            # Multi-timeframe analysis
            try:
                analysis.update(self._calculate_mtf_analysis(mtf_data))
            except Exception as e:
                logger.error(f"Error calculating multi-timeframe analysis for {symbol}: {e}")
            
            # Generate signals
            try:
                analysis.update(self._generate_technical_signals(analysis, mtf_data))
            except Exception as e:
                logger.error(f"Error generating technical signals for {symbol}: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced technical analysis for {symbol}: {e}")
            return {}
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        
        # RSI
        rsi = self.momentum.calculate_rsi(close, 14)
        
        # ATR
        atr = self._calculate_atr(high, low, close, 14)
        
        return {
            "sma_20": sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
            "sma_50": sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
            "ema_12": ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else None,
            "ema_26": ema_26.iloc[-1] if not pd.isna(ema_26.iloc[-1]) else None,
            "rsi_14": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            "atr_14": atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None,
            "current_price": close.iloc[-1]
        }
    
    def _calculate_momentum_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced momentum indicators"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # RSI and divergence
        rsi = self.momentum.calculate_rsi(close, 14)
        rsi_divergence = self.momentum.detect_rsi_divergence(close, rsi)
        
        # MACD
        macd_data = self.momentum.calculate_macd(close)
        
        # Stochastic RSI
        stoch_rsi_data = self.momentum.calculate_stochastic_rsi(close)
        
        # Williams %R
        williams_r = self.momentum.calculate_williams_r(high, low, close, 14)
        
        # ROC
        roc_data = self.momentum.calculate_roc(close, [5, 10, 20])
        
        return {
            "rsi_divergence": rsi_divergence,
            "macd": macd_data["macd"].iloc[-1] if not pd.isna(macd_data["macd"].iloc[-1]) else None,
            "macd_signal": macd_data["signal"].iloc[-1] if not pd.isna(macd_data["signal"].iloc[-1]) else None,
            "macd_histogram": macd_data["histogram"].iloc[-1] if not pd.isna(macd_data["histogram"].iloc[-1]) else None,
            "stoch_rsi_k": stoch_rsi_data["stoch_rsi_k"].iloc[-1] if not pd.isna(stoch_rsi_data["stoch_rsi_k"].iloc[-1]) else None,
            "stoch_rsi_d": stoch_rsi_data["stoch_rsi_d"].iloc[-1] if not pd.isna(stoch_rsi_data["stoch_rsi_d"].iloc[-1]) else None,
            "williams_r": williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else None,
            "roc_5": roc_data["roc_5"].iloc[-1] if not pd.isna(roc_data["roc_5"].iloc[-1]) else None,
            "roc_10": roc_data["roc_10"].iloc[-1] if not pd.isna(roc_data["roc_10"].iloc[-1]) else None,
            "roc_20": roc_data["roc_20"].iloc[-1] if not pd.isna(roc_data["roc_20"].iloc[-1]) else None
        }
    
    def _calculate_volume_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile and analysis"""
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # VWAP and bands
        vwap = self.volume.calculate_vwap(high, low, close, volume)
        vwap_bands = self.volume.calculate_vwap_bands(vwap, volume)
        
        # OBV and A/D Line
        obv = self.volume.calculate_obv(close, volume)
        ad_line = self.volume.calculate_ad_line(high, low, close, volume)
        
        # Volume profile
        volume_profile = self.volume.calculate_volume_profile(high, low, volume)
        
        return {
            "vwap": vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else None,
            "vwap_upper": vwap_bands["vwap_upper"].iloc[-1] if not pd.isna(vwap_bands["vwap_upper"].iloc[-1]) else None,
            "vwap_lower": vwap_bands["vwap_lower"].iloc[-1] if not pd.isna(vwap_bands["vwap_lower"].iloc[-1]) else None,
            "obv": obv.iloc[-1] if not pd.isna(obv.iloc[-1]) else None,
            "ad_line": ad_line.iloc[-1] if not pd.isna(ad_line.iloc[-1]) else None,
            "volume_profile": volume_profile
        }
    
    def _calculate_mtf_analysis(self, mtf_data: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Any]:
        """Calculate multi-timeframe analysis"""
        mtf_analysis = {}
        
        for tf_name, data in mtf_data.items():
            if data is None or data.empty or len(data) < 10:
                continue
                
            close = data['Close']
            
            # Calculate trend direction for each timeframe
            if len(close) >= 20:
                sma_20 = close.rolling(20).mean()
                trend = "bullish" if close.iloc[-1] > sma_20.iloc[-1] else "bearish"
            else:
                trend = "neutral"
            
            # Calculate momentum
            if len(close) >= 5:
                momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
            else:
                momentum = 0
            
            mtf_analysis[f"{tf_name}_trend"] = trend
            mtf_analysis[f"{tf_name}_momentum"] = momentum
        
        return mtf_analysis
    
    def _generate_technical_signals(self, analysis: Dict[str, Any], mtf_data: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Any]:
        """Generate technical signals based on analysis"""
        signals = {
            "momentum_signals": [],
            "volume_signals": [],
            "divergence_signals": [],
            "overall_signal": "neutral"
        }
        
        # Momentum signals
        rsi = analysis.get("rsi_14", 50)
        if rsi < 30:
            signals["momentum_signals"].append("oversold")
        elif rsi > 70:
            signals["momentum_signals"].append("overbought")
        
        # MACD signals
        macd = analysis.get("macd", 0)
        macd_signal = analysis.get("macd_signal", 0)
        if macd > macd_signal:
            signals["momentum_signals"].append("macd_bullish")
        else:
            signals["momentum_signals"].append("macd_bearish")
        
        # Divergence signals
        if analysis.get("rsi_divergence", {}).get("bullish_divergence"):
            signals["divergence_signals"].append("bullish_divergence")
        if analysis.get("rsi_divergence", {}).get("bearish_divergence"):
            signals["divergence_signals"].append("bearish_divergence")
        
        # Volume signals
        current_price = analysis.get("current_price", 0)
        vwap = analysis.get("vwap", 0)
        if current_price > vwap:
            signals["volume_signals"].append("above_vwap")
        else:
            signals["volume_signals"].append("below_vwap")
        
        # Overall signal determination
        bullish_count = len([s for s in signals["momentum_signals"] if "bullish" in s or s == "oversold"])
        bearish_count = len([s for s in signals["momentum_signals"] if "bearish" in s or s == "overbought"])
        
        if bullish_count > bearish_count:
            signals["overall_signal"] = "bullish"
        elif bearish_count > bullish_count:
            signals["overall_signal"] = "bearish"
        
        return signals
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr

# Singleton instances
enhanced_technical = EnhancedTechnicalFramework()
