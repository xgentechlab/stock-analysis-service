"""
FastAPI route for identifying hot stocks using yfinance over the last 7 days.
Criteria:
- Volume spike (>1.5x 20-day average)
- Price momentum (>5% move over last 5 trading days)
- Institutional activity (proxy via yfinance fund holders/major holders presence)
Filter to large-cap universe (top 200) and return 30-40 stocks.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import uuid

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import ApiResponse, HotStockSelection, HotStocksRunMetadata, HotStocksRun
from app.services.stocks import stocks_service
from app.models.schemas import JobCreateRequest, AnalysisType
from app.services.analysis_trigger import analysis_trigger
from app.db.firestore_client import firestore_client

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["hot-stocks"]) 

# Minimal mapping to fix common Yahoo Finance NSE ticker differences
YAHOO_SYMBOL_MAP: Dict[str, str] = {
    # Company common names to Yahoo tickers (without .NS)
    "INFOSYS": "INFY",
    "MCDOWELL": "MCDOWELL-N",
}


def _normalize_to_yahoo(symbol: str) -> str:
    """Normalize common NSE symbols to Yahoo-compatible tickers (no suffix)."""
    base = symbol.replace(".NS", "")
    mapped = YAHOO_SYMBOL_MAP.get(base, base)
    return mapped

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

def _format_enhanced_technical_indicators(technical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format technical indicators for storage (matches stage processor format)"""
    return {
        "basic_indicators": {
            "sma_20": technical_data.get("sma_20"),
            "sma_50": technical_data.get("sma_50"),
            "rsi_14": technical_data.get("rsi_14"),
            "atr_14": technical_data.get("atr_14")
        },
        "momentum_indicators": {
            "macd": technical_data.get("macd"),
            "macd_signal": technical_data.get("macd_signal"),
            "macd_histogram": technical_data.get("macd_histogram"),
            "stoch_rsi_k": technical_data.get("stoch_rsi_k"),
            "stoch_rsi_d": technical_data.get("stoch_rsi_d"),
            "williams_r": technical_data.get("williams_r"),
            "roc_5": technical_data.get("roc_5"),
            "roc_10": technical_data.get("roc_10"),
            "roc_20": technical_data.get("roc_20")
        },
        "volume_indicators": {
            "vwap": technical_data.get("vwap"),
            "vwap_upper": technical_data.get("vwap_upper"),
            "vwap_lower": technical_data.get("vwap_lower"),
            "obv": technical_data.get("obv"),
            "ad_line": technical_data.get("ad_line")
        },
        "divergence_signals": {
            "bullish_divergence": technical_data.get("bullish_divergence", False),
            "bearish_divergence": technical_data.get("bearish_divergence", False)
        },
        "multi_timeframe": {
            "1m_trend": technical_data.get("1m_trend"),
            "5m_trend": technical_data.get("5m_trend"),
            "15m_trend": technical_data.get("15m_trend"),
            "1d_trend": technical_data.get("1d_trend"),
            "1wk_trend": technical_data.get("1wk_trend")
        }
    }


def _compute_signals_for_symbol(symbol: str, use_enhanced_indicators: bool = True) -> Dict[str, Any]:
    """
    Compute signal metrics for a single symbol using yfinance.
    Returns dict with metrics or empty dict on failure.
    """
    try:
        logger.debug(f"[signal-compute] Starting signal computation for {symbol}")
        
        # Ensure NSE suffix for yfinance with normalization
        normalized = _normalize_to_yahoo(symbol)
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

        # Base signals
        signals = {
            "symbol": symbol,
            "momentum_pct_5d": float(price_momentum_pct),
            "volume_spike_ratio": float(volume_spike_ratio),
            "institutional": bool(inst_flag),
            "fundamentals": fundamental_data,
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
                
                # Add enhanced indicators to signals
                signals.update({
                    "rsi": technical_data.get('rsi14'),
                    "macd": technical_data.get('macd'),
                    "macd_signal": technical_data.get('macd_signal'),
                    "macd_histogram": technical_data.get('macd_histogram'),
                    "stochastic_k": technical_data.get('stoch_k'),
                    "stochastic_d": technical_data.get('stoch_d'),
                    "williams_r": technical_data.get('williams_r'),
                    "bb_upper": technical_data.get('bb_upper'),
                    "bb_middle": technical_data.get('bb_middle'),
                    "bb_lower": technical_data.get('bb_lower'),
                    "bb_position": _calculate_bb_position(technical_data),
                    "obv": technical_data.get('obv'),
                    "ema12": technical_data.get('ema12'),
                    "ema26": technical_data.get('ema26'),
                    "trend_score": trend_score,
                    "momentum_osc_score": momentum_osc_score,
                    "volume_momentum_score": volume_momentum_score,
                    "composite_score": composite_score,
                    "fundamental_score": fundamental_score
                })
                
                # ENHANCED ANALYSIS: Add Stage 1 & 2 functionality from analysis pipeline
                try:
                    logger.debug(f"[signal-compute] {symbol}: Computing Stage 1 & 2 enhanced analysis")
                    from app.services.stocks import stocks_service
                    
                    # STAGE 1: Enhanced Data Collection and Analysis
                    enhanced_stock_info = stocks_service.get_enhanced_stock_info(symbol)
                    if enhanced_stock_info:
                        # Use data already fetched in enhanced_stock_info (optimized single-pass)
                        enhanced_technical_data = enhanced_stock_info.get('enhanced_technical', {})
                        enhanced_technical_scores = enhanced_stock_info.get('enhanced_technical_scores', {})
                        enhanced_fundamentals = enhanced_stock_info.get('enhanced_fundamentals', {})
                        enhanced_fundamental_score_data = enhanced_stock_info.get('fundamental_score', {})
                        
                        # STAGE 2: Enhanced Technical and Combined Scoring
                        if enhanced_technical_data and enhanced_technical_scores:
                            # Use pre-calculated technical scores (optimized)
                            enhanced_fundamental_score = enhanced_fundamental_score_data.get('final_score', 0.0)
                            enhanced_technical_score = enhanced_technical_scores.get('final_score', 0.0)
                            
                            # Enhanced combined score (same weights as stage processor)
                            enhanced_combined_score = (0.6 * enhanced_technical_score + 0.4 * enhanced_fundamental_score)
                            
                            # Format technical indicators like stage processor
                            formatted_indicators = _format_enhanced_technical_indicators(enhanced_technical_data)
                            
                            # Add enhanced analysis to signals
                            signals.update({
                                # Enhanced technical indicators (structured like stage processor)
                                "enhanced_technical_indicators": formatted_indicators,
                                
                                # Enhanced scores
                                "enhanced_technical_score": enhanced_technical_score,
                                "enhanced_fundamental_score": enhanced_fundamental_score,
                                "enhanced_combined_score": enhanced_combined_score,
                                "enhanced_technical_confidence": enhanced_technical_scores.get('confidence', 0.0),
                                "enhanced_technical_strength": enhanced_technical_scores.get('strength', 'unknown'),
                                
                                # Enhanced fundamental data
                                "enhanced_fundamentals": enhanced_fundamentals,
                                
                                # Stage 1 metadata
                                "stage_1_metadata": {
                                    "indicators_available": len(enhanced_technical_data) if enhanced_technical_data else 0,
                                    "status": "success" if enhanced_technical_data else "failed",
                                    "summary": {
                                        "momentum_indicators": len([k for k in formatted_indicators.get('momentum_indicators', {}).keys()]),
                                        "volume_indicators": len([k for k in formatted_indicators.get('volume_indicators', {}).keys()]),
                                        "trend_indicators": len([k for k in formatted_indicators.get('basic_indicators', {}).keys()]),
                                        "oscillator_indicators": len([k for k in formatted_indicators.get('momentum_indicators', {}).keys() if 'rsi' in k.lower() or 'stoch' in k.lower()])
                                    }
                                },
                                
                                # Stage 2 metadata
                                "stage_2_metadata": {
                                    "technical_score": enhanced_technical_score,
                                    "fundamental_score": enhanced_fundamental_score,
                                    "combined_score": enhanced_combined_score,
                                    "scoring_weights": {
                                        "technical": 0.6,
                                        "fundamental": 0.4
                                    }
                                },
                                
                                # Multi-timeframe analysis ID (if available)
                                "multi_timeframe_analysis_id": enhanced_stock_info.get('multi_timeframe_analysis_id', ''),
                                
                                # Data quality indicators
                                "data_fetch_optimized": enhanced_stock_info.get('data_fetch_optimized', False)
                            })
                            
                            logger.debug(f"[signal-compute] {symbol}: Enhanced analysis completed - tech={enhanced_technical_score:.3f} fund={enhanced_fundamental_score:.3f} combined={enhanced_combined_score:.3f}")
                        else:
                            logger.debug(f"[signal-compute] {symbol}: Enhanced technical analysis not available")
                    else:
                        logger.debug(f"[signal-compute] {symbol}: Enhanced stock info not available")
                        
                except Exception as e:
                    logger.warning(f"[signal-compute] {symbol}: Enhanced Stage 1 & 2 analysis failed: {e}")
                    # Continue with existing enhanced indicators
                
            except Exception as e:
                logger.warning(f"[signal-compute] {symbol}: Enhanced indicators calculation failed: {e}")
                # Continue with base signals only
        
        logger.debug(f"[signal-compute] {symbol}: Signal computation completed successfully")
        return signals
    except Exception as e:
        logger.warning(f"[signal-compute] {symbol}: Signal computation failed: {e}")
        return {}


def _store_hot_stock_analysis(symbol: str, analysis_data: Dict[str, Any]) -> str:
    """Store hot stock analysis data in the database"""
    try:
        from app.db.firestore_client import firestore_client
        from datetime import datetime, timezone
        import uuid
        
        # Create analysis document
        analysis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Extract technical indicators
        technical_indicators = {
            "rsi": analysis_data.get("rsi"),
            "macd": analysis_data.get("macd"),
            "macd_signal": analysis_data.get("macd_signal"),
            "macd_histogram": analysis_data.get("macd_histogram"),
            "stochastic_k": analysis_data.get("stochastic_k"),
            "stochastic_d": analysis_data.get("stochastic_d"),
            "williams_r": analysis_data.get("williams_r"),
            "bb_upper": analysis_data.get("bb_upper"),
            "bb_middle": analysis_data.get("bb_middle"),
            "bb_lower": analysis_data.get("bb_lower"),
            "bb_position": analysis_data.get("bb_position"),
            "obv": analysis_data.get("obv"),
            "ema12": analysis_data.get("ema12"),
            "ema26": analysis_data.get("ema26"),
            # Enhanced technical indicators from Stage 1 & 2 (structured)
            "enhanced_technical_indicators": analysis_data.get("enhanced_technical_indicators", {})
        }
        
        # Extract scores
        scores = {
            "trend_score": analysis_data.get("trend_score"),
            "momentum_osc_score": analysis_data.get("momentum_osc_score"),
            "volume_momentum_score": analysis_data.get("volume_momentum_score"),
            "composite_score": analysis_data.get("composite_score"),
            "fundamental_score": analysis_data.get("fundamental_score"),
            "momentum_pct_5d": analysis_data.get("momentum_pct_5d"),
            "volume_spike_ratio": analysis_data.get("volume_spike_ratio"),
            # Enhanced scores from Stage 1 & 2
            "enhanced_technical_score": analysis_data.get("enhanced_technical_score"),
            "enhanced_fundamental_score": analysis_data.get("enhanced_fundamental_score"),
            "enhanced_combined_score": analysis_data.get("enhanced_combined_score"),
            "enhanced_technical_confidence": analysis_data.get("enhanced_technical_confidence"),
            "enhanced_technical_strength": analysis_data.get("enhanced_technical_strength")
        }
        
        # Create hot stock analysis document
        hot_stock_analysis = {
            "analysis_id": analysis_id,
            "symbol": symbol,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "analysis_type": "hot_stock_selection",
            "technical_indicators": technical_indicators,
            "fundamentals": analysis_data.get("fundamentals", {}),
            "enhanced_fundamentals": analysis_data.get("enhanced_fundamentals", {}),
            "scores": scores,
            "institutional_activity": analysis_data.get("institutional", False),
            "selection_rank": 0,  # Will be set by caller
            "analysis_version": "1.0",
            "data_quality": "good",
            # Enhanced analysis metadata
            "multi_timeframe_analysis_id": analysis_data.get("multi_timeframe_analysis_id", ""),
            "data_fetch_optimized": analysis_data.get("data_fetch_optimized", False),
            "stage_1_2_integrated": True,
            # Stage-specific metadata (matches stage processor format)
            "stage_1_metadata": analysis_data.get("stage_1_metadata", {}),
            "stage_2_metadata": analysis_data.get("stage_2_metadata", {})
        }
        
        # Store in database
        firestore_client.create_hot_stock_analysis(hot_stock_analysis)
        
        logger.info(f"Stored hot stock analysis {analysis_id} for {symbol}")
        return analysis_id
        
    except Exception as e:
        logger.error(f"Error storing hot stock analysis for {symbol}: {e}")
        raise


@router.get("/hot-stocks", response_model=ApiResponse)
async def get_hot_stocks(
    limit: int = Query(10, ge=1, le=60, description="Max number of stocks to return"),
    universe_size: int = Query(50, ge=1, le=500, description="Universe size (top-N stocks)"),
    market_cap_tier: str = Query("all", description="Market cap tier: all, large_cap, mid_cap, small_cap"),
    min_momentum_pct: float = Query(0.5, description=">= momentum threshold over ~5 days"),
    min_volume_spike: float = Query(0.05, description=">= volume spike ratio vs 20D avg"),
    require_institutional: bool = Query(False, description="Require institutional activity proxy"),
    use_enhanced_indicators: bool = Query(True, description="Use enhanced technical indicators"),
    max_pe_ratio: float = Query(50.0, description="Maximum P/E ratio allowed"),
    min_roe: float = Query(10.0, description="Minimum ROE percentage"),
    min_market_cap_cr: float = Query(1000.0, description="Minimum market cap in crores"),
):
    """Return hot stocks from expanded universe using enhanced signals."""
    # Initialize run tracking
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.now()
    start_time = time.time()
    
    try:
        # Use expanded universe with market cap filtering
        logger.info(f"[hot-stocks] Fetching universe (size={universe_size}, tier={market_cap_tier})")
        universe: List[str] = stocks_service.get_expanded_universe_symbols(
            limit=universe_size, 
            market_cap_tier=market_cap_tier
        )
        logger.info(f"[hot-stocks] Universe fetched: count={len(universe)} sample={universe[:5]}")

        # Compute signals for each symbol
        metrics: List[Dict[str, Any]] = []
        logger.info(f"[hot-stocks] Processing {len(universe)} stocks from universe")
        
        for i, sym in enumerate(universe, 1):
            logger.info(f"[hot-stocks] Processing stock {i}/{len(universe)}: {sym}")
            m = _compute_signals_for_symbol(sym, use_enhanced_indicators)
            if m:
                metrics.append(m)
                logger.info(f"[hot-stocks] ✅ {sym}: mom_5d={m.get('momentum_pct_5d', 0):.2f}% vol_spike={m.get('volume_spike_ratio', 0):.2f}x inst={m.get('institutional')} composite={m.get('composite_score', 0):.2f}")
            else:
                logger.warning(f"[hot-stocks] ❌ {sym}: Failed to compute signals")

        if not metrics:
            logger.info("[hot-stocks] No metrics computed; returning empty list")
            return ApiResponse(ok=True, data={"hot_stocks": [], "total": 0})

        # Apply filters
        logger.info(f"[hot-stocks] Applying filters: min_momentum_pct={min_momentum_pct}%, min_volume_spike={min_volume_spike}x, require_institutional={require_institutional}, max_pe={max_pe_ratio}, min_roe={min_roe}%, min_mcap={min_market_cap_cr}cr")
        filtered = []
        
        for m in metrics:
            symbol = m.get("symbol", "UNKNOWN")
            momentum = m.get("momentum_pct_5d", 0.0)
            volume_spike = m.get("volume_spike_ratio", 0.0)
            institutional = m.get("institutional", False)
            fundamentals = m.get("fundamentals", {})
            
            # Extract fundamental metrics
            pe_ratio = fundamentals.get("pe_ratio")
            roe = fundamentals.get("roe")
            # Convert ROE from decimal percentage to percentage (0.1 -> 10%)
            roe_percentage = roe * 100 if roe is not None else None
            market_cap = fundamentals.get("market_cap", 0)
            market_cap_cr = market_cap / 10000000 if market_cap else 0  # Convert to crores
            
            roe_display = f"{roe_percentage:.2f}" if roe_percentage is not None else "N/A"
            logger.info(f"[hot-stocks] Filtering {symbol}: mom={momentum:.2f}% vol={volume_spike:.2f}x inst={institutional} pe={pe_ratio} roe={roe_display}% mcap={market_cap_cr:.0f}cr")
            
            # Check momentum filter
            if momentum < min_momentum_pct:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - momentum {momentum:.2f}% < {min_momentum_pct}%")
                continue
            
            # Check volume filter
            if volume_spike < min_volume_spike:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - volume spike {volume_spike:.2f}x < {min_volume_spike}x")
                continue
            
            # Check institutional filter
            if require_institutional and not institutional:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - no institutional activity")
                continue
            
            # Check fundamental filters
            if pe_ratio and pe_ratio > max_pe_ratio:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - P/E {pe_ratio:.1f} > {max_pe_ratio}")
                continue
            
            if roe_percentage and roe_percentage < min_roe:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - ROE {roe_percentage:.2f}% < {min_roe}%")
                continue
            
            if market_cap_cr < min_market_cap_cr:
                logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - market cap {market_cap_cr:.0f}cr < {min_market_cap_cr}cr")
                continue
            
            # Stock passed all filters
            logger.info(f"[hot-stocks] ✅ {symbol}: ACCEPTED - passed all filters")
            filtered.append(m)
        
        logger.info(f"[hot-stocks] Filtering complete: kept={len(filtered)} of {len(metrics)} stocks")

        # Rank by combined score: prioritize enhanced Stage 1 & 2 scores, fallback to basic scores
        def score(m: Dict[str, Any]) -> float:
            # PRIORITY 1: Use enhanced Stage 1 & 2 combined score if available
            enhanced_combined_score = m.get("enhanced_combined_score")
            if enhanced_combined_score is not None:
                # Use enhanced combined score with momentum and volume boosters
                momentum_boost = min(0.2, abs(m.get("momentum_pct_5d", 0.0)) / 50.0)  # Max 0.2 boost
                volume_boost = min(0.1, (m.get("volume_spike_ratio", 1.0) - 1.0) / 5.0)  # Max 0.1 boost
                enhanced_score = enhanced_combined_score + momentum_boost + volume_boost
                return enhanced_score
            
            # PRIORITY 2: Use basic composite score if enhanced not available
            elif use_enhanced_indicators and "composite_score" in m:
                # Use composite technical score as primary, with momentum and volume as boosters
                composite = m.get("composite_score", 0.0)
                fundamental = m.get("fundamental_score", 0.5)  # Default neutral fundamental score
                momentum_boost = min(0.2, abs(m.get("momentum_pct_5d", 0.0)) / 50.0)  # Max 0.2 boost
                volume_boost = min(0.1, (m.get("volume_spike_ratio", 1.0) - 1.0) / 5.0)  # Max 0.1 boost
                
                # Combined score: 60% technical + 40% fundamental + boosters
                combined_score = (0.6 * composite + 0.4 * fundamental) + momentum_boost + volume_boost
                return combined_score
            else:
                # Fallback to original scoring
                return m.get("momentum_pct_5d", 0.0) + 20.0 * (m.get("volume_spike_ratio", 0.0) - 1.0)

        # Rank stocks by score
        logger.info(f"[hot-stocks] Ranking {len(filtered)} stocks by score")
        ranked = sorted(filtered, key=score, reverse=True)
        
        # Log ranking details
        for i, stock in enumerate(ranked[:5], 1):  # Log top 5
            symbol = stock.get("symbol", "UNKNOWN")
            stock_score = score(stock)
            momentum = stock.get("momentum_pct_5d", 0.0)
            volume_spike = stock.get("volume_spike_ratio", 0.0)
            
            # Show enhanced scores if available, otherwise basic scores
            enhanced_combined = stock.get("enhanced_combined_score")
            if enhanced_combined is not None:
                enhanced_tech = stock.get("enhanced_technical_score", 0.0)
                enhanced_fund = stock.get("enhanced_fundamental_score", 0.0)
                logger.info(f"[hot-stocks] Rank {i}: {symbol} - score={stock_score:.3f} (ENHANCED: tech={enhanced_tech:.3f} fund={enhanced_fund:.3f} combined={enhanced_combined:.3f} mom={momentum:.2f}% vol={volume_spike:.2f}x)")
            else:
                composite = stock.get("composite_score", 0.0)
                fundamental = stock.get("fundamental_score", 0.5)
                logger.info(f"[hot-stocks] Rank {i}: {symbol} - score={stock_score:.3f} (BASIC: tech={composite:.3f} fund={fundamental:.3f} mom={momentum:.2f}% vol={volume_spike:.2f}x)")
        
        top_n = ranked[:limit]
        logger.info(f"[hot-stocks] Selected top {len(top_n)} stocks (limit={limit})")
        
        if len(top_n) > 0:
            top_symbols = [x.get("symbol") for x in top_n]
            logger.info(f"[hot-stocks] Final selection: {top_symbols}")
        else:
            logger.warning("[hot-stocks] No stocks selected after ranking")

        # Store hot stock analysis data in database
        stored_count = 0
        for item in top_n:
            sym = item.get("symbol")
            try:
                logger.info(f"[hot-stocks] Storing analysis data for hot stock: {sym}")
                _store_hot_stock_analysis(sym, item)
                stored_count += 1
            except Exception as e:
                logger.warning(f"[hot-stocks] Failed to store analysis data for {sym}: {e}")
        
        logger.info(f"[hot-stocks] Analysis data stored: {stored_count}/{len(top_n)}")

        # Auto-trigger analysis for returned hot stocks (fire-and-forget)
        triggered: int = 0
        for item in top_n:
            sym = item.get("symbol")
            try:
                logger.info("[hot-stocks] Triggering analysis for hot stock | symbol=%s", sym)
                req = JobCreateRequest(symbol=sym, analysis_type=AnalysisType.ENHANCED)
                # Use cache by default; set force_refresh=False to avoid extra cost
                logger.info(f"Triggering analysis for {sym}")
                #analysis_trigger.fire_and_forget(req, force_refresh=False)
                triggered += 1
            except Exception as te:
                logger.warning(f"[hot-stocks] Failed to trigger analysis for {sym}: {te}")
        logger.info(f"[hot-stocks] Analysis jobs triggered: {triggered}/{len(top_n)}")

        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create hot stock selections for storage
        hot_stock_selections = []
        for i, stock in enumerate(top_n, 1):
            fundamentals = stock.get("fundamentals", {})
            market_cap = fundamentals.get("market_cap", 0)
            market_cap_cr = market_cap / 10000000 if market_cap else 0
            
            selection = HotStockSelection(
                symbol=stock.get("symbol", ""),
                rank=i,
                enhanced_technical_score=stock.get("enhanced_technical_score"),
                enhanced_fundamental_score=stock.get("enhanced_fundamental_score"),
                enhanced_combined_score=stock.get("enhanced_combined_score"),
                basic_composite_score=stock.get("composite_score"),
                basic_fundamental_score=stock.get("fundamental_score"),
                momentum_pct_5d=stock.get("momentum_pct_5d", 0.0),
                volume_spike_ratio=stock.get("volume_spike_ratio", 0.0),
                institutional_activity=stock.get("institutional", False),
                pe_ratio=fundamentals.get("pe_ratio"),
                roe=fundamentals.get("roe"),
                market_cap_cr=market_cap_cr,
                sector=fundamentals.get("sector"),
                industry=fundamentals.get("industry"),
                analysis_id=stock.get("analysis_id", ""),
                multi_timeframe_analysis_id=stock.get("multi_timeframe_analysis_id")
            )
            hot_stock_selections.append(selection)

        # Create run metadata
        metadata = HotStocksRunMetadata(
            run_id=run_id,
            run_timestamp=run_timestamp,
            universe_size=len(universe),
            total_processed=len(metrics),
            total_filtered=len(filtered),
            total_selected=len(top_n),
            processing_time_seconds=processing_time,
            filters_applied={
                "min_momentum_pct": min_momentum_pct,
                "min_volume_spike": min_volume_spike,
                "require_institutional": require_institutional,
                "max_pe_ratio": max_pe_ratio,
                "min_roe": min_roe,
                "min_market_cap_cr": min_market_cap_cr,
                "market_cap_tier": market_cap_tier
            },
            selection_criteria={
                "limit": limit,
                "universe_size": universe_size,
                "use_enhanced_indicators": use_enhanced_indicators
            },
            data_quality="good",
            stage_1_2_integrated=True,
            data_fetch_optimized=True,
            api_version="1.0"
        )

        # Create hot stocks run
        hot_stocks_run = HotStocksRun(
            run_id=run_id,
            run_timestamp=run_timestamp,
            metadata=metadata,
            hot_stocks=hot_stock_selections,
            summary={
                "total_universe": len(universe),
                "total_processed": len(metrics),
                "total_filtered": len(filtered),
                "total_selected": len(top_n),
                "processing_time_seconds": processing_time,
                "triggered_jobs": triggered,
                "enhanced_scoring_used": any(s.get("enhanced_combined_score") is not None for s in top_n)
            },
            created_at=run_timestamp,
            updated_at=run_timestamp
        )

        # Store the run in Firestore
        try:
            run_data = hot_stocks_run.dict()
            stored_run_id = firestore_client.create_hot_stocks_run(run_data)
            logger.info(f"[hot-stocks] Stored hot stocks run: {stored_run_id}")
        except Exception as e:
            logger.error(f"[hot-stocks] Failed to store hot stocks run: {e}")

        return ApiResponse(
            ok=True,
            data={
                "hot_stocks": top_n,
                "total": len(top_n),
                "triggered_jobs": triggered,
                "run_id": run_id,
                "processing_time_seconds": processing_time,
            },
        )
    except Exception as e:
        logger.error(f"Failed to compute hot stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-stocks/runs", response_model=ApiResponse)
async def get_hot_stocks_runs(
    limit: int = Query(10, ge=1, le=100, description="Number of runs to return"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get hot stocks runs with optional date filtering."""
    try:
        runs = firestore_client.list_hot_stocks_runs(
            limit=limit,
            start_date=start_date,
            end_date=end_date
        )
        
        return ApiResponse(
            ok=True,
            data={
                "runs": runs,
                "total": len(runs)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get hot stocks runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-stocks/runs/latest", response_model=ApiResponse)
async def get_latest_hot_stocks_run():
    """Get the latest hot stocks run."""
    try:
        logger.info("[hot-stocks-runs] Getting latest hot stocks run")
        
        # First try to list runs to see what's available
        all_runs = firestore_client.list_hot_stocks_runs(limit=5)
        logger.info(f"[hot-stocks-runs] Found {len(all_runs)} total runs")
        
        # Then try the latest query
        run = firestore_client.get_latest_hot_stocks_run()
        logger.info(f"[hot-stocks-runs] Latest query result: {run is not None}")
        
        if not run:
            logger.warning("[hot-stocks-runs] No hot stocks runs found")
            # If latest query fails, try to get the first from the list
            if all_runs:
                logger.info("[hot-stocks-runs] Using first run from list as fallback")
                run = all_runs[0]
            else:
                raise HTTPException(status_code=404, detail="No hot stocks runs found")
        
        logger.info(f"[hot-stocks-runs] Found latest run: {run.get('run_id', 'unknown')}")
        return ApiResponse(
            ok=True,
            data=run
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest hot stocks run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-stocks/runs/{run_id}", response_model=ApiResponse)
async def get_hot_stocks_run(run_id: str):
    """Get a specific hot stocks run by ID."""
    try:
        run = firestore_client.get_hot_stocks_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Hot stocks run not found")
        
        return ApiResponse(
            ok=True,
            data=run
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hot stocks run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/hot-stocks/runs/{run_id}", response_model=ApiResponse)
async def delete_hot_stocks_run(run_id: str):
    """Delete a hot stocks run."""
    try:
        success = firestore_client.delete_hot_stocks_run(run_id)
        if not success:
            raise HTTPException(status_code=404, detail="Hot stocks run not found")
        
        return ApiResponse(
            ok=True,
            data={"message": f"Hot stocks run {run_id} deleted successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete hot stocks run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


