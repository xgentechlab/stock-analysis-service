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

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window).mean()

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

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    std = prices.rolling(window).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {"upper": upper, "middle": sma, "lower": lower}

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return {"k_percent": k_percent, "d_percent": d_percent}

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    price_change = close.diff()
    obv = volume.copy()
    obv[price_change < 0] = -volume[price_change < 0]
    obv[price_change == 0] = 0
    return obv.cumsum()

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
        
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns for technical analysis: {missing_columns}")
            return {}
        
        close = ohlcv_data['Close']
        high = ohlcv_data['High']
        low = ohlcv_data['Low']
        volume = ohlcv_data['Volume']
        
        # Calculate indicators
        sma20 = calculate_sma(close, 20)
        sma50 = calculate_sma(close, 50)
        ema12 = calculate_ema(close, 12)
        ema26 = calculate_ema(close, 26)
        rsi14 = calculate_rsi(close, 14)
        atr14 = calculate_atr(high, low, close, 14)
        vol20 = calculate_volume_average(volume, 20)
        
        # Calculate new indicators with error handling
        try:
            macd_data = calculate_macd(close)
        except Exception as e:
            logger.warning(f"Failed to calculate MACD: {e}")
            macd_data = {"macd": pd.Series([0]), "signal": pd.Series([0]), "histogram": pd.Series([0])}
        
        try:
            bb_data = calculate_bollinger_bands(close)
        except Exception as e:
            logger.warning(f"Failed to calculate Bollinger Bands: {e}")
            bb_data = {"upper": pd.Series([0]), "middle": pd.Series([0]), "lower": pd.Series([0])}
        
        try:
            stoch_data = calculate_stochastic(high, low, close)
        except Exception as e:
            logger.warning(f"Failed to calculate Stochastic: {e}")
            stoch_data = {"k_percent": pd.Series([50]), "d_percent": pd.Series([50])}
        
        try:
            williams_r = calculate_williams_r(high, low, close)
        except Exception as e:
            logger.warning(f"Failed to calculate Williams %R: {e}")
            williams_r = pd.Series([-50])
        
        try:
            obv = calculate_obv(close, volume)
        except Exception as e:
            logger.warning(f"Failed to calculate OBV: {e}")
            obv = pd.Series([0])
        
        # Price changes
        pct_1d, pct_5d = calculate_price_changes(close)
        
        # Current values (latest)
        current_close = close.iloc[-1]
        current_sma20 = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else None
        current_sma50 = sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else None
        current_ema12 = ema12.iloc[-1] if not pd.isna(ema12.iloc[-1]) else None
        current_ema26 = ema26.iloc[-1] if not pd.isna(ema26.iloc[-1]) else None
        current_rsi = rsi14.iloc[-1] if not pd.isna(rsi14.iloc[-1]) else None
        current_atr = atr14.iloc[-1] if not pd.isna(atr14.iloc[-1]) else None
        current_vol20 = int(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else None
        current_vol_today = int(volume.iloc[-1])
        
        # New indicator values
        current_macd = macd_data["macd"].iloc[-1] if not pd.isna(macd_data["macd"].iloc[-1]) else None
        current_macd_signal = macd_data["signal"].iloc[-1] if not pd.isna(macd_data["signal"].iloc[-1]) else None
        current_macd_histogram = macd_data["histogram"].iloc[-1] if not pd.isna(macd_data["histogram"].iloc[-1]) else None
        current_bb_upper = bb_data["upper"].iloc[-1] if not pd.isna(bb_data["upper"].iloc[-1]) else None
        current_bb_middle = bb_data["middle"].iloc[-1] if not pd.isna(bb_data["middle"].iloc[-1]) else None
        current_bb_lower = bb_data["lower"].iloc[-1] if not pd.isna(bb_data["lower"].iloc[-1]) else None
        current_stoch_k = stoch_data["k_percent"].iloc[-1] if not pd.isna(stoch_data["k_percent"].iloc[-1]) else None
        current_stoch_d = stoch_data["d_percent"].iloc[-1] if not pd.isna(stoch_data["d_percent"].iloc[-1]) else None
        current_williams_r = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else None
        current_obv = obv.iloc[-1] if not pd.isna(obv.iloc[-1]) else None
        
        # Breakout check
        breakout = is_breakout(close, high, 20)
        
        return {
            "close": float(current_close),
            "sma20": float(current_sma20) if current_sma20 is not None else None,
            "sma50": float(current_sma50) if current_sma50 is not None else None,
            "ema12": float(current_ema12) if current_ema12 is not None else None,
            "ema26": float(current_ema26) if current_ema26 is not None else None,
            "rsi14": float(current_rsi) if current_rsi is not None else None,
            "atr14": float(current_atr) if current_atr is not None else None,
            "vol20": int(current_vol20) if current_vol20 is not None else None,
            "vol_today": int(current_vol_today),
            "pct_1d": float(pct_1d),
            "pct_5d": float(pct_5d),
            "is_breakout": bool(breakout),  # Convert numpy.bool_ to Python bool
            # New indicators
            "macd": float(current_macd) if current_macd is not None else None,
            "macd_signal": float(current_macd_signal) if current_macd_signal is not None else None,
            "macd_histogram": float(current_macd_histogram) if current_macd_histogram is not None else None,
            "bb_upper": float(current_bb_upper) if current_bb_upper is not None else None,
            "bb_middle": float(current_bb_middle) if current_bb_middle is not None else None,
            "bb_lower": float(current_bb_lower) if current_bb_lower is not None else None,
            "stoch_k": float(current_stoch_k) if current_stoch_k is not None else None,
            "stoch_d": float(current_stoch_d) if current_stoch_d is not None else None,
            "williams_r": float(current_williams_r) if current_williams_r is not None else None,
            "obv": float(current_obv) if current_obv is not None else None
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

def calculate_trend_score(technical_data: Dict[str, Optional[float]]) -> float:
    """Calculate trend strength score based on moving averages"""
    try:
        close = technical_data.get('close', 0.0)
        sma20 = technical_data.get('sma20')
        sma50 = technical_data.get('sma50')
        ema12 = technical_data.get('ema12')
        ema26 = technical_data.get('ema26')
        
        score = 0.0
        
        # SMA trend
        if sma20 is not None and sma50 is not None and close is not None:
            if close > sma20 > sma50:
                score += 0.4
            elif close > sma20:
                score += 0.2
        
        # EMA trend
        if ema12 is not None and ema26 is not None and close is not None:
            if close > ema12 > ema26:
                score += 0.4
            elif close > ema12:
                score += 0.2
        
        return min(1.0, score)
    except Exception as e:
        logger.error(f"Error calculating trend score: {e}")
        return 0.0

def calculate_momentum_oscillator_score(technical_data: Dict[str, Optional[float]]) -> float:
    """Calculate momentum score from RSI, Stochastic, Williams %R"""
    try:
        rsi = technical_data.get('rsi14')
        stoch_k = technical_data.get('stoch_k')
        stoch_d = technical_data.get('stoch_d')
        williams_r = technical_data.get('williams_r')
        
        # Use default values if None
        rsi = rsi if rsi is not None else 50.0
        stoch_k = stoch_k if stoch_k is not None else 50.0
        stoch_d = stoch_d if stoch_d is not None else 50.0
        williams_r = williams_r if williams_r is not None else -50.0
        
        score = 0.0
        
        # RSI momentum
        if 30 <= rsi <= 70:  # Not overbought/oversold
            score += 0.3
        elif 50 < rsi <= 70:  # Bullish momentum
            score += 0.4
        
        # Stochastic momentum
        if 20 <= stoch_k <= 80 and stoch_k > stoch_d:  # Bullish crossover
            score += 0.3
        
        # Williams %R momentum
        if -80 <= williams_r <= -20:  # Not extreme
            score += 0.2
        elif -50 <= williams_r <= -20:  # Bullish
            score += 0.3
        
        return min(1.0, score)
    except Exception as e:
        logger.error(f"Error calculating momentum oscillator score: {e}")
        return 0.0

def calculate_volume_momentum_score(technical_data: Dict[str, Optional[float]]) -> float:
    """Calculate volume momentum using OBV and volume patterns"""
    try:
        obv = technical_data.get('obv')
        vol_today = technical_data.get('vol_today')
        vol20 = technical_data.get('vol20')
        
        # Use default values if None
        obv = obv if obv is not None else 0.0
        vol_today = vol_today if vol_today is not None else 0
        vol20 = vol20 if vol20 is not None else 1
        
        score = 0.0
        
        # Volume spike
        if vol_today and vol20 and vol_today > vol20 * 1.5:
            score += 0.6
        
        # OBV trend (simplified - would need historical OBV for proper trend)
        if obv > 0:
            score += 0.4
        
        return min(1.0, score)
    except Exception as e:
        logger.error(f"Error calculating volume momentum score: {e}")
        return 0.0

def calculate_composite_technical_score(technical_data: Dict[str, Optional[float]]) -> float:
    """Calculate composite technical score combining all indicators"""
    try:
        # Get individual scores
        momentum_score = calculate_momentum_score(technical_data)
        volume_score = calculate_volume_spike_score(technical_data)
        breakout_score = calculate_breakout_volatility_score(technical_data)
        trend_score = calculate_trend_score(technical_data)
        momentum_osc_score = calculate_momentum_oscillator_score(technical_data)
        volume_momentum_score = calculate_volume_momentum_score(technical_data)
        
        # Weighted combination
        weights = {
            'momentum': 0.2,
            'volume': 0.2,
            'breakout': 0.15,
            'trend': 0.2,
            'momentum_osc': 0.15,
            'volume_momentum': 0.1
        }
        
        composite = (
            weights['momentum'] * momentum_score +
            weights['volume'] * volume_score +
            weights['breakout'] * breakout_score +
            weights['trend'] * trend_score +
            weights['momentum_osc'] * momentum_osc_score +
            weights['volume_momentum'] * volume_momentum_score
        )
        
        return min(1.0, composite)
    except Exception as e:
        logger.error(f"Error calculating composite technical score: {e}")
        return 0.0
