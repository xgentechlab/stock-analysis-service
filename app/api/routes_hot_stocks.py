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
from app.analysis.utilities.compute_signals import _compute_signals_for_symbol
from app.analysis.utilities.store_hot_stocks import _store_hot_stock_analysis, _store_hot_stocks_run
from app.analysis.utilities.stock_filtering import get_filtered_stocks, score_stock


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["hot-stocks"]) 

# Minimal mapping to fix common Yahoo Finance NSE ticker differences
YAHOO_SYMBOL_MAP: Dict[str, str] = {
    # Company common names to Yahoo tickers (without .NS)
    "INFOSYS": "INFY",
    "MCDOWELL": "MCDOWELL-N",
}








@router.get("/hot-stocks", response_model=ApiResponse)
async def get_hot_stocks(
    limit: int = Query(100, ge=1, le=100, description="Max number of stocks to return"),
    universe_size: int = Query(50, ge=1, le=500, description="Universe size (top-N stocks)"),
    market_cap_tier: str = Query("all", description="Market cap tier: all, large_cap, mid_cap, small_cap"),
    min_momentum_pct: float = Query(0.5, description=">= momentum threshold over ~5 days"),
    min_volume_spike: float = Query(0.05, description=">= volume spike ratio vs 20D avg"),
    require_institutional: bool = Query(False, description="Require institutional activity proxy"),
    use_enhanced_indicators: bool = Query(True, description="Use enhanced technical indicators"),
    max_pe_ratio: float = Query(30.0, description="Maximum P/E ratio allowed"),
    min_roe: float = Query(12.0, description="Minimum ROE percentage"),
    min_market_cap_cr: float = Query(1000.0, description="Minimum market cap in crores"),
    max_debt_equity: float = Query(0.5, description="Maximum debt-to-equity ratio"),
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
        logger.info(f"[hot-stocks] Universe fetched: count={len(universe)} ")

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
          
            if get_filtered_stocks(m, min_momentum_pct, min_volume_spike, require_institutional, max_pe_ratio, min_roe, min_market_cap_cr, max_debt_equity):
                filtered.append(m)
                logger.info(f"[hot-stocks] ✅ {m.get('symbol', 'UNKNOWN')}: ACCEPTED - passed all filters")
            else:
                logger.info(f"[hot-stocks] ❌ {m.get('symbol', 'UNKNOWN')}: REJECTED - did not pass all filters")
        
        logger.info(f"[hot-stocks] Filtering complete: kept={len(filtered)} of {len(metrics)} stocks")

        
        # Rank stocks by score
        logger.info(f"[hot-stocks] Ranking {len(filtered)} stocks by score")
        ranked = sorted(filtered, key=lambda x: score_stock(x, use_enhanced_indicators), reverse=True)
        
        
        top_n = ranked[:limit]
        logger.info(f"[hot-stocks] Selected top {len(top_n)} stocks (limit={limit})")
        
        if len(top_n) > 0:
            top_symbols = [x.get("symbol") for x in top_n]
            logger.info(f"[hot-stocks] Final selection: {top_symbols}")
        else:
            logger.warning("[hot-stocks] No stocks selected after ranking")

        # Store hot stock analysis data in database
        _store_hot_stock_analysis(top_n)

        # Auto-trigger analysis for returned hot stocks (fire-and-forget)
        triggered: int = 0
        for item in top_n:
            sym = item.get("symbol")
            try:
                logger.info("[hot-stocks] Triggering analysis for hot stock | symbol=%s", sym)
                req = JobCreateRequest(symbol=sym, analysis_type=AnalysisType.ENHANCED)
                # Use cache by default; set force_refresh=False to avoid extra cost
                logger.info(f"Triggering analysis for {sym}")
                analysis_trigger.fire_and_forget(req, force_refresh=True)
                triggered += 1
            except Exception as te:
                logger.warning(f"[hot-stocks] Failed to trigger analysis for {sym}: {te}")
        logger.info(f"[hot-stocks] Analysis jobs triggered: {triggered}/{len(top_n)}")

        # Calculate processing time
        processing_time = time.time() - start_time
        
        _store_hot_stocks_run(top_n, run_id, run_timestamp, processing_time, triggered, universe, metrics, filtered, min_momentum_pct, min_volume_spike, require_institutional, max_pe_ratio, min_roe, min_market_cap_cr, max_debt_equity, market_cap_tier, limit, universe_size, use_enhanced_indicators)

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


