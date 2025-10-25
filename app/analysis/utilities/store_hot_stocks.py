import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import HotStockSelection, HotStocksRunMetadata, HotStocksRun
from app.db.firestore_client import firestore_client
from datetime import datetime, timezone
import uuid
logger = logging.getLogger(__name__)


def _store_hot_stock_analysis(top_n: List[Dict[str, Any]]) -> None:
    """Store hot stock analysis data in the database"""
    stored_count = 0
    for item in top_n:
        sym = item.get("symbol")
        try:
            logger.info(f"[hot-stocks] Storing analysis data for hot stock: {sym}")
            _process_hot_stock_analysis(sym, item)
            stored_count += 1
        except Exception as e:
            logger.warning(f"[hot-stocks] Failed to store analysis data for {sym}: {e}")
    
    logger.info(f"[hot-stocks] Analysis data stored: {stored_count}/{len(top_n)}")
    


def _process_hot_stock_analysis(symbol: str, analysis_data: Dict[str, Any]) -> str:
    """Process hot stock analysis data"""
    try:
        from app.db.firestore_client import firestore_client
        from datetime import datetime, timezone
        import uuid
        
        # Create analysis document
        analysis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Extract technical indicators (consolidated format only)
        technical_indicators = {
            # Use only enhanced technical indicators (structured format)
            "enhanced_technical_indicators": analysis_data.get("enhanced_technical_indicators", {})
        }
        
        # Extract scores (consolidated format only)
        scores = {
            # Essential signals
            "momentum_pct_5d": analysis_data.get("momentum_pct_5d"),
            "volume_spike_ratio": analysis_data.get("volume_spike_ratio"),
            
            # Enhanced scores (primary scoring system)
            "enhanced_technical_score": analysis_data.get("enhanced_technical_score"),
            "enhanced_fundamental_score": analysis_data.get("enhanced_fundamental_score"),
            "enhanced_combined_score": analysis_data.get("enhanced_combined_score"),
            "enhanced_technical_confidence": analysis_data.get("enhanced_technical_confidence"),
            "enhanced_technical_strength": analysis_data.get("enhanced_technical_strength"),
            
            # Component scores (for debugging/analysis)
            "trend_score": analysis_data.get("trend_score"),
            "momentum_osc_score": analysis_data.get("momentum_osc_score"),
            "volume_momentum_score": analysis_data.get("volume_momentum_score")
        }
        
        # Create hot stock analysis document (optimized)
        hot_stock_analysis = {
            "analysis_id": analysis_id,
            "symbol": symbol,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "analysis_type": "hot_stock_selection",
            "technical_indicators": technical_indicators,
            "enhanced_fundamentals": analysis_data.get("enhanced_fundamentals", {}),
            "scores": scores,
            "institutional_activity": analysis_data.get("institutional", False),
            "selection_rank": 0,  # Will be set by caller
            "analysis_version": "2.0",  # Updated version
            "data_quality": "good",
            # Enhanced analysis metadata
            "multi_timeframe_analysis_id": analysis_data.get("multi_timeframe_analysis_id", ""),
            "data_fetch_optimized": analysis_data.get("data_fetch_optimized", False),
            "optimized_structure": True  # Flag for optimized data structure
        }
        
        # Store in database
        firestore_client.create_hot_stock_analysis(hot_stock_analysis)
        
        logger.info(f"Stored hot stock analysis {analysis_id} for {symbol}")
        return analysis_id
        
    except Exception as e:
        logger.error(f"Error storing hot stock analysis for {symbol}: {e}")
        raise


# Store the run in Firestore
def _store_hot_stocks_run(top_n: List[Dict[str, Any]], run_id: str, run_timestamp: datetime, processing_time: float, triggered: int, 
                         universe: List[str] = None, metrics: List[Dict[str, Any]] = None, filtered: List[Dict[str, Any]] = None,
                         min_momentum_pct: float = 0.1, min_volume_spike: float = 0.01, require_institutional: bool = False,
                         max_pe_ratio: float = 100.0, min_roe: float = 5.0, min_market_cap_cr: float = 500.0,
                         market_cap_tier: str = "all", limit: int = 10, universe_size: int = 50, 
                         use_enhanced_indicators: bool = True) -> None:
    """Store the run in Firestore"""
    logger.info(f"[hot-stocks] Storing run with {len(top_n)} stocks: {[s.get('symbol') for s in top_n]}")
    hot_stock_selections = []
    for i, stock in enumerate(top_n, 1):
        symbol = stock.get("symbol", "UNKNOWN")
        logger.info(f"[hot-stocks] Processing stock {i}: {symbol}")
        
        fundamentals = stock.get("fundamentals", {})
        market_cap = fundamentals.get("market_cap", 0)
        market_cap_cr = market_cap / 10000000 if market_cap else 0
        
        logger.info(f"[hot-stocks] {symbol} - Enhanced scores: tech={stock.get('enhanced_technical_score')}, fund={stock.get('enhanced_fundamental_score')}, combined={stock.get('enhanced_combined_score')}")
        logger.info(f"[hot-stocks] {symbol} - Basic scores: composite={stock.get('composite_score')}, fundamental={stock.get('fundamental_score')}")
        
        selection = HotStockSelection(
            symbol=stock.get("symbol", ""),
            rank=i,
            enhanced_technical_score=stock.get("enhanced_technical_score"),
            enhanced_fundamental_score=stock.get("enhanced_fundamental_score"),
            enhanced_combined_score=stock.get("enhanced_combined_score"),
            basic_composite_score=stock.get("composite_score") or stock.get("enhanced_technical_score"),  # Fallback to enhanced
            basic_fundamental_score=stock.get("fundamental_score") or stock.get("enhanced_fundamental_score"),  # Fallback to enhanced
            momentum_pct_5d=stock.get("momentum_pct_5d", 0.0),
            volume_spike_ratio=stock.get("volume_spike_ratio", 0.0),
            institutional_activity=stock.get("institutional", False),
            pe_ratio=fundamentals.get("pe_ratio"),
            roe=fundamentals.get("roe"),
            market_cap_cr=market_cap_cr,
            sector=fundamentals.get("sector"),
            industry=fundamentals.get("industry"),
            analysis_id=stock.get("analysis_id", ""),  # Will be populated by later stages
            multi_timeframe_analysis_id=stock.get("multi_timeframe_analysis_id")
        )
        hot_stock_selections.append(selection)

    # Create run metadata
    metadata = HotStocksRunMetadata(
        run_id=run_id,
        run_timestamp=run_timestamp,
        universe_size=len(universe) if universe else 0,
        total_processed=len(metrics) if metrics else 0,
        total_filtered=len(filtered) if filtered else 0,
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
            # "total_universe": len(universe),
            # "total_processed": len(metrics),
            # "total_filtered": len(filtered),
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
        run_data = hot_stocks_run.model_dump()  # Use model_dump instead of dict()
        logger.info(f"[hot-stocks] Run data prepared with {len(run_data.get('hot_stocks', []))} stocks")
        stored_run_id = firestore_client.create_hot_stocks_run(run_data)
        logger.info(f"[hot-stocks] Stored hot stocks run: {stored_run_id}")
    except Exception as e:
        logger.error(f"[hot-stocks] Error storing run: {e}")
        import traceback
        logger.error(f"[hot-stocks] Traceback: {traceback.format_exc()}")
        raise
    