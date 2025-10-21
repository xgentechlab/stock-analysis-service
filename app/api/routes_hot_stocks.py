"""
FastAPI route for identifying hot stocks using yfinance over the last 7 days.
Criteria:
- Volume spike (>1.5x 20-day average)
- Price momentum (>5% move over last 5 trading days)
- Institutional activity (proxy via yfinance fund holders/major holders presence)
Filter to large-cap universe (top 200) and return 30-40 stocks.
"""
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import ApiResponse
from app.services.stocks import stocks_service
from app.models.schemas import JobCreateRequest, AnalysisType
from app.services.analysis_trigger import analysis_trigger

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


def _compute_signals_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Compute signal metrics for a single symbol using yfinance.
    Returns dict with metrics or empty dict on failure.
    """
    try:
        # Ensure NSE suffix for yfinance with normalization
        normalized = _normalize_to_yahoo(symbol)
        ticker_symbol = normalized if normalized.endswith('.NS') else f"{normalized}.NS"
        ticker = yf.Ticker(ticker_symbol)

        # Fetch last ~40 trading days to compute 20D average reliably
        hist = ticker.history(period="2mo")
        if hist is None or hist.empty:
            return {}

        # Use only necessary columns
        df = hist[["Close", "Volume"]].dropna().copy()
        if len(df) < 25:
            return {}

        # Momentum over last 5 trading days (approx a week)
        recent = df.tail(5)
        if recent.empty:
            return {}
        price_momentum_pct = (recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1.0) * 100.0

        # Volume spike: compare last day's volume to 20-day average
        vol_20d_avg = df["Volume"].tail(20).mean()
        last_vol = df["Volume"].iloc[-1]
        volume_spike_ratio = float(last_vol / vol_20d_avg) if vol_20d_avg and vol_20d_avg > 0 else 0.0

        # Institutional activity proxy: if major holders or fund holders data is present
        inst_flag = False
        try:
            major_holders = ticker.major_holders
            fund_holders = ticker.funds
            inst_flag = (major_holders is not None and len(major_holders) > 0) or (
                fund_holders is not None and len(fund_holders) > 0
            )
        except Exception:
            # If holders calls fail, keep flag as False
            inst_flag = False

        return {
            "symbol": symbol,
            "momentum_pct_5d": float(price_momentum_pct),
            "volume_spike_ratio": float(volume_spike_ratio),
            "institutional": bool(inst_flag),
        }
    except Exception as e:
        logger.warning(f"Signal computation failed for {symbol}: {e}")
        return {}


@router.get("/hot-stocks", response_model=ApiResponse)
async def get_hot_stocks(
    limit: int = Query(40, ge=10, le=60, description="Max number of stocks to return"),
    universe_size: int = Query(200, ge=50, le=200, description="Universe size (top-N large-caps)"),
    min_momentum_pct: float = Query(5.0, description=">= momentum threshold over ~5 days"),
    min_volume_spike: float = Query(1.5, description=">= volume spike ratio vs 20D avg"),
    require_institutional: bool = Query(False, description="Require institutional activity proxy"),
):
    """Return 30-40 hot stocks from large-cap universe using simple signals."""
    try:
        # Use configured large-cap universe
        logger.info(f"[hot-stocks] Fetching universe (size={universe_size})")
        universe: List[str] = stocks_service.get_universe_symbols(limit=universe_size)
        logger.info(f"[hot-stocks] Universe fetched: count={len(universe)} sample={universe[:5]}")

        # Compute signals for each symbol
        metrics: List[Dict[str, Any]] = []
        for sym in universe:
            m = _compute_signals_for_symbol(sym)
            if m:
                metrics.append(m)
                if len(metrics) <= 5:
                    logger.debug(
                        f"[hot-stocks] metric {sym}: mom_5d={m.get('momentum_pct_5d'):.2f} vol_spike={m.get('volume_spike_ratio'):.2f} inst={m.get('institutional')}"
                    )

        if not metrics:
            logger.info("[hot-stocks] No metrics computed; returning empty list")
            return ApiResponse(ok=True, data={"hot_stocks": [], "total": 0})

        # Apply filters
        logger.info(
            f"[hot-stocks] Applying filters: min_momentum_pct={min_momentum_pct}, min_volume_spike={min_volume_spike}, require_institutional={require_institutional}"
        )
        filtered = []
        for m in metrics:
            if m.get("momentum_pct_5d", 0.0) < min_momentum_pct:
                continue
            if m.get("volume_spike_ratio", 0.0) < min_volume_spike:
                continue
            if require_institutional and not m.get("institutional", False):
                continue
            filtered.append(m)
        logger.info(f"[hot-stocks] After filters: kept={len(filtered)} of {len(metrics)}")

        # Rank by combined score: normalize simple sum of z-less metrics
        def score(m: Dict[str, Any]) -> float:
            return m.get("momentum_pct_5d", 0.0) + 20.0 * (m.get("volume_spike_ratio", 0.0) - 1.0)

        ranked = sorted(filtered, key=score, reverse=True)
        top_n = ranked[:limit]
        logger.info(f"[hot-stocks] Returning top {len(top_n)} (limit={limit})")
        top_symbols = [x.get("symbol") for x in top_n[:10]]
        logger.debug("[hot-stocks] Top symbols: %s", top_symbols)

        # Auto-trigger analysis for returned hot stocks (fire-and-forget)
        triggered: int = 0
        for item in top_n:
            sym = item.get("symbol")
            try:
                logger.info("[hot-stocks] Triggering analysis for hot stock | symbol=%s", sym)
                req = JobCreateRequest(symbol=sym, analysis_type=AnalysisType.ENHANCED)
                # Use cache by default; set force_refresh=False to avoid extra cost
                analysis_trigger.fire_and_forget(req, force_refresh=False)
                triggered += 1
            except Exception as te:
                logger.warning(f"[hot-stocks] Failed to trigger analysis for {sym}: {te}")
        logger.info(f"[hot-stocks] Analysis jobs triggered: {triggered}/{len(top_n)}")

        return ApiResponse(
            ok=True,
            data={
                "hot_stocks": top_n,
                "total": len(top_n),
                "triggered_jobs": triggered,
            },
        )
    except Exception as e:
        logger.error(f"Failed to compute hot stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


