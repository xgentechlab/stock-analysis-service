"""
Technical indicators calculation module
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_volume_average(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volume moving average"""
    return volume.rolling(window=window).mean()

def calculate_price_changes(close: pd.Series) -> Tuple[float, float]:
    """Calculate 1-day and 5-day price changes"""
    if len(close) < 6:
        return 0.0, 0.0
    
    pct_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0.0
    pct_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) >= 6 else 0.0
    
    return pct_1d, pct_5d

def is_breakout(close: pd.Series, high: pd.Series, window: int = 20) -> bool:
    """Check if current price is breaking out of recent high"""
    if len(high) < window:
        return False
    
    recent_high = high.iloc[-(window+1):-1].max()  # Exclude current day
    current_close = close.iloc[-1]
    
    return current_close > recent_high

def calculate_technical_snapshot(ohlcv_data: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate all technical indicators for a given OHLCV DataFrame
    
    Expected DataFrame columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        if len(ohlcv_data) < 30:  # Reduced from 60 to 30 days minimum
            logger.warning(f"Insufficient data for technical analysis: {len(ohlcv_data)} days")
            return {}
        
        close = ohlcv_data['Close']
        high = ohlcv_data['High']
        low = ohlcv_data['Low']
        volume = ohlcv_data['Volume']
        
        # Calculate indicators
        sma20 = calculate_sma(close, 20)
        sma50 = calculate_sma(close, 50)
        rsi14 = calculate_rsi(close, 14)
        atr14 = calculate_atr(high, low, close, 14)
        vol20 = calculate_volume_average(volume, 20)
        
        # Price changes
        pct_1d, pct_5d = calculate_price_changes(close)
        
        # Current values (latest)
        current_close = close.iloc[-1]
        current_sma20 = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else None
        current_sma50 = sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else None
        current_rsi = rsi14.iloc[-1] if not pd.isna(rsi14.iloc[-1]) else None
        current_atr = atr14.iloc[-1] if not pd.isna(atr14.iloc[-1]) else None
        current_vol20 = int(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else None
        current_vol_today = int(volume.iloc[-1])
        
        # Breakout check
        breakout = is_breakout(close, high, 20)
        
        return {
            "close": float(current_close),
            "sma20": float(current_sma20) if current_sma20 is not None else None,
            "sma50": float(current_sma50) if current_sma50 is not None else None,
            "rsi14": float(current_rsi) if current_rsi is not None else None,
            "atr14": float(current_atr) if current_atr is not None else None,
            "vol20": int(current_vol20) if current_vol20 is not None else None,
            "vol_today": int(current_vol_today),
            "pct_1d": float(pct_1d),
            "pct_5d": float(pct_5d),
            "is_breakout": bool(breakout)  # Convert numpy.bool_ to Python bool
        }
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {}

def calculate_momentum_score(technical_data: Dict[str, Optional[float]], max_expected: float = 0.10) -> float:
    """
    Calculate momentum score (0-1)
    Uses 1-day and 5-day price changes, plus SMA crossover
    """
    try:
        pct_1d = technical_data.get('pct_1d', 0.0)
        pct_5d = technical_data.get('pct_5d', 0.0)
        close = technical_data.get('close', 0.0)
        sma20 = technical_data.get('sma20')
        sma50 = technical_data.get('sma50')
        
        # Basic momentum from price changes
        m1 = (pct_1d + pct_5d * 0.5) / max_expected
        m1 = max(-1, min(1, m1))  # Clamp to [-1, 1]
        momentum_base = (m1 + 1) / 2  # Convert to [0, 1]
        momentum_base = float(momentum_base)  # Ensure Python float
        
        # SMA trend boost
        sma_boost = 0.0
        if sma20 and sma50 and close > sma20 and sma20 > sma50:
            sma_boost = 0.2
        
        final_score = min(1.0, momentum_base + sma_boost)
        return float(final_score)  # Ensure Python float
        
    except Exception as e:
        logger.error(f"Error calculating momentum score: {e}")
        return 0.0

def calculate_volume_spike_score(technical_data: Dict[str, Optional[float]], 
                               threshold: float = 2.0, cap: float = 5.0) -> float:
    """
    Calculate volume spike score (0-1)
    """
    try:
        vol_today = technical_data.get('vol_today', 0)
        vol20 = technical_data.get('vol20', 1)
        
        if not vol_today or not vol20 or vol20 == 0:
            return 0.0
        
        volume_ratio = vol_today / vol20
        
        if volume_ratio >= cap:
            return 1.0
        elif volume_ratio <= 1.0:
            return 0.0
        else:
            return float((volume_ratio - 1) / (cap - 1))  # Ensure Python float
        
    except Exception as e:
        logger.error(f"Error calculating volume spike score: {e}")
        return 0.0

def calculate_breakout_volatility_score(technical_data: Dict[str, Optional[float]]) -> float:
    """
    Calculate breakout/volatility score (0-1)
    Combines breakout signal with normalized ATR
    """
    try:
        is_breakout = technical_data.get('is_breakout', False)
        atr14 = technical_data.get('atr14', 0.0)
        close = technical_data.get('close', 1.0)
        
        # Breakout component (0 or 1)
        breakout_component = 1.0 if is_breakout else 0.0
        
        # Volatility component (normalized ATR)
        if atr14 and close and close > 0:
            normalized_atr = atr14 / close
            # Map to 0-1 range (assuming ATR/Close ratio of 0.05 = high volatility)
            volatility_component = min(1.0, normalized_atr / 0.05)
        else:
            volatility_component = 0.0
        
        # Weighted combination
        final_score = 0.7 * breakout_component + 0.3 * volatility_component
        return float(min(1.0, final_score))  # Ensure Python float
        
    except Exception as e:
        logger.error(f"Error calculating breakout volatility score: {e}")
        return 0.0
