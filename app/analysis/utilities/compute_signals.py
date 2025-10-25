import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import yfinance as yf
from app.analysis.utilities.data_formatters import _format_enhanced_technical_indicators


logger = logging.getLogger(__name__)


def _calculate_bb_position(technical_data: Dict[str, Optional[float]]) -> str:
    """Calculate Bollinger Bands position"""
    close = technical_data.get('close', 0)
    bb_upper = technical_data.get('bb_upper')
    bb_lower = technical_data.get('bb_lower')
    
    if not all([close, bb_upper, bb_lower]):
        return "unknown"
    
    if close > bb_upper:
        return "above_upper"
    elif close < bb_lower:
        return "below_lower"
    else:
        return "within_bands"




def _compute_signals_for_symbol(symbol: str, use_enhanced_indicators: bool = True) -> Dict[str, Any]:
    """
    Compute signal metrics for a single symbol using yfinance.
    Returns dict with metrics or empty dict on failure.
    """
    try:
        logger.debug(f"[signal-compute] Starting signal computation for {symbol}")
        
        # Ensure NSE suffix for yfinance with normalization
        normalized = symbol
        ticker_symbol = normalized if normalized.endswith('.NS') else f"{normalized}.NS"
        logger.debug(f"[signal-compute] {symbol} -> {ticker_symbol}")
        
        ticker = yf.Ticker(ticker_symbol)

        # Fetch last ~40 trading days to compute 20D average reliably
        logger.debug(f"[signal-compute] {symbol}: Fetching historical data")
        hist = ticker.history(period="2mo")
        if hist is None or hist.empty:
            logger.warning(f"[signal-compute] {symbol}: No historical data available")
            return {}

        # Use all OHLCV columns for technical analysis
        df = hist[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 25:
            logger.warning(f"[signal-compute] {symbol}: Insufficient data ({len(df)} days < 25)")
            return {}
        
        logger.debug(f"[signal-compute] {symbol}: Got {len(df)} days of data")

        # Momentum over last 5 trading days (approx a week)
        recent = df.tail(5)
        if recent.empty:
            logger.warning(f"[signal-compute] {symbol}: No recent data for momentum calculation")
            return {}
        
        price_start = recent["Close"].iloc[0]
        price_end = recent["Close"].iloc[-1]
        price_momentum_pct = (price_end / price_start - 1.0) * 100.0
        logger.debug(f"[signal-compute] {symbol}: Price momentum: {price_start:.2f} -> {price_end:.2f} = {price_momentum_pct:.2f}%")

        # Volume spike: compare last day's volume to 20-day average
        vol_20d_avg = df["Volume"].tail(20).mean()
        last_vol = df["Volume"].iloc[-1]
        volume_spike_ratio = float(last_vol / vol_20d_avg) if vol_20d_avg and vol_20d_avg > 0 else 0.0
        
        # Additional volume analysis
        vol_5d_avg = df["Volume"].tail(5).mean()
        vol_10d_avg = df["Volume"].tail(10).mean()
        vol_max_20d = df["Volume"].tail(20).max()
        vol_min_20d = df["Volume"].tail(20).min()
        
        logger.debug(f"[signal-compute] {symbol}: Volume analysis:")
        logger.debug(f"[signal-compute] {symbol}:   Last day: {last_vol:,.0f}")
        logger.debug(f"[signal-compute] {symbol}:   5-day avg: {vol_5d_avg:,.0f}")
        logger.debug(f"[signal-compute] {symbol}:   10-day avg: {vol_10d_avg:,.0f}")
        logger.debug(f"[signal-compute] {symbol}:   20-day avg: {vol_20d_avg:,.0f}")
        logger.debug(f"[signal-compute] {symbol}:   20-day range: {vol_min_20d:,.0f} - {vol_max_20d:,.0f}")
        logger.debug(f"[signal-compute] {symbol}:   Spike ratio: {volume_spike_ratio:.2f}x")

        # Institutional activity proxy: if major holders or fund holders data is present
        inst_flag = False
        try:
            logger.debug(f"[signal-compute] {symbol}: Checking institutional activity")
            major_holders = ticker.major_holders
            fund_holders = ticker.funds
            inst_flag = (major_holders is not None and len(major_holders) > 0) or (
                fund_holders is not None and len(fund_holders) > 0
            )
            logger.debug(f"[signal-compute] {symbol}: Institutional activity: {inst_flag}")
        except Exception as e:
            # If holders calls fail, keep flag as False
            logger.debug(f"[signal-compute] {symbol}: Institutional check failed: {e}")
            inst_flag = False

        # Fetch fundamental data
        fundamental_data = {}
        try:
            logger.debug(f"[signal-compute] {symbol}: Fetching fundamental data")
            info = ticker.info
            if info:
                # Try multiple ROE field names as yfinance might use different ones
                roe = (info.get("returnOnEquity") or 
                       info.get("returnOnEquityTTM") or 
                       info.get("returnOnEquityQuarterly") or
                       info.get("roe") or
                       None)
                
                fundamental_data = {
                    "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "roe": roe,
                    "eps_ttm": info.get("trailingEps"),
                    "market_cap": info.get("marketCap"),
                    "dividend_yield": info.get("dividendYield"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry")
                }
                logger.debug(f"[signal-compute] {symbol}: Fundamental data - PE={fundamental_data.get('pe_ratio')}, ROE={fundamental_data.get('roe')}, MarketCap={fundamental_data.get('market_cap')}")
                
                # Debug ROE fields for troubleshooting
                if symbol == "ITC" and not roe:
                    roe_fields = {
                        "returnOnEquity": info.get("returnOnEquity"),
                        "returnOnEquityTTM": info.get("returnOnEquityTTM"),
                        "returnOnEquityQuarterly": info.get("returnOnEquityQuarterly"),
                        "roe": info.get("roe")
                    }
                    logger.debug(f"[signal-compute] {symbol}: ROE fields debug: {roe_fields}")
            else:
                logger.warning(f"[signal-compute] {symbol}: No fundamental data available")
        except Exception as e:
            logger.warning(f"[signal-compute] {symbol}: Fundamental data fetch failed: {e}")
            fundamental_data = {}

        # Base signals (essential data only)
        signals = {
            "symbol": symbol,
            "current_price": price_end,  # NON-NEGOTIABLE: Store current price
            "momentum_pct_5d": float(price_momentum_pct),
            "volume_spike_ratio": float(volume_spike_ratio),
            "institutional": bool(inst_flag),
            "fundamentals": fundamental_data,  # Keep basic fundamentals for fallback
        }
        
        # Add enhanced indicators if requested
        if use_enhanced_indicators:
            try:
                logger.debug(f"[signal-compute] {symbol}: Computing enhanced indicators")
                from app.services.indicators import (
                    calculate_technical_snapshot,
                    calculate_trend_score,
                    calculate_momentum_oscillator_score,
                    calculate_volume_momentum_score,
                    calculate_composite_technical_score
                )
                
                # Calculate comprehensive technical snapshot
                technical_data = calculate_technical_snapshot(df)
                logger.debug(f"[signal-compute] {symbol}: Technical snapshot computed with {len(technical_data)} indicators")
                
                # Calculate additional scores
                trend_score = calculate_trend_score(technical_data)
                momentum_osc_score = calculate_momentum_oscillator_score(technical_data)
                volume_momentum_score = calculate_volume_momentum_score(technical_data)
                composite_score = calculate_composite_technical_score(technical_data)
                
                logger.debug(f"[signal-compute] {symbol}: Enhanced scores - trend={trend_score:.3f} momentum_osc={momentum_osc_score:.3f} volume_mom={volume_momentum_score:.3f} composite={composite_score:.3f}")
                
                # Calculate fundamental score if fundamental data is available
                fundamental_score = 0.5  # Default neutral score
                if fundamental_data:
                    try:
                        from app.services.fundamental_scoring import FundamentalScoringEngine
                        fundamental_scorer = FundamentalScoringEngine()
                        fundamental_score_data = fundamental_scorer.calculate_fundamental_score(fundamental_data)
                        if fundamental_score_data and "final_score" in fundamental_score_data:
                            fundamental_score = fundamental_score_data["final_score"]
                            logger.debug(f"[signal-compute] {symbol}: Fundamental score calculated: {fundamental_score:.3f}")
                        else:
                            logger.debug(f"[signal-compute] {symbol}: Using default fundamental score: {fundamental_score}")
                    except Exception as e:
                        logger.warning(f"[signal-compute] {symbol}: Enhanced fundamental scoring failed: {e}")
                        fundamental_score = 0.5
                else:
                    logger.debug(f"[signal-compute] {symbol}: No fundamental data for enhanced scoring")
                
                # Add enhanced indicators to signals (consolidated format)
                signals.update({
                    # Technical indicators in structured format (no duplicates)
                    "enhanced_technical_indicators": _format_enhanced_technical_indicators(technical_data),
                    
                    # Consolidated scores (enhanced only)
                    "enhanced_technical_score": composite_score,  # Use composite as technical score
                    "enhanced_fundamental_score": fundamental_score,
                    "enhanced_combined_score": (0.6 * composite_score + 0.4 * fundamental_score),
                    "enhanced_technical_confidence": 0.8,  # Default confidence
                    "enhanced_technical_strength": "medium" if composite_score > 0.5 else "weak",
                    
                    # Individual component scores (for debugging/analysis)
                    "trend_score": trend_score,
                    "momentum_osc_score": momentum_osc_score,
                    "volume_momentum_score": volume_momentum_score,
                })
                
                # Add enhanced fundamental data if available
                try:
                    logger.debug(f"[signal-compute] {symbol}: Fetching enhanced fundamental data")
                    from app.services.stocks import stocks_service
                    
                    enhanced_stock_info = stocks_service.get_enhanced_stock_info(symbol)
                    if enhanced_stock_info:
                        enhanced_fundamentals = enhanced_stock_info.get('enhanced_fundamentals', {})
                        if enhanced_fundamentals:
                            signals["enhanced_fundamentals"] = enhanced_fundamentals
                            logger.debug(f"[signal-compute] {symbol}: Enhanced fundamentals added")
                        
                        # Add multi-timeframe analysis ID if available
                        multi_timeframe_id = enhanced_stock_info.get('multi_timeframe_analysis_id', '')
                        if multi_timeframe_id:
                            signals["multi_timeframe_analysis_id"] = multi_timeframe_id
                            signals["data_fetch_optimized"] = enhanced_stock_info.get('data_fetch_optimized', False)
                    else:
                        logger.debug(f"[signal-compute] {symbol}: Enhanced stock info not available")
                        
                except Exception as e:
                    logger.warning(f"[signal-compute] {symbol}: Enhanced fundamental data fetch failed: {e}")
                    # Continue without enhanced fundamentals
                
            except Exception as e:
                logger.warning(f"[signal-compute] {symbol}: Enhanced indicators calculation failed: {e}")
                # Continue with base signals only
        
        logger.debug(f"[signal-compute] {symbol}: Signal computation completed successfully")
        return signals
    except Exception as e:
        logger.warning(f"[signal-compute] {symbol}: Signal computation failed: {e}")
        return {}
