"""
FastAPI routes for admin and cron endpoints
"""
import math
from fastapi import APIRouter, HTTPException
from typing import Optional, Any, Dict

from app.models.schemas import ApiResponse, RuntimeConfig, SelectionSummary, TrackerSummary
from app.db.firestore_client import firestore_client
from app.models.firestore_models import dict_to_runtime_config, runtime_config_to_dict
from app.services.selection import selection_engine
from app.services.tracker import position_tracker
from app.services.openai_client import openai_client
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def _sanitize_json_data(data: Any) -> Any:
    """Recursively sanitize data to ensure JSON compliance"""
    if isinstance(data, dict):
        return {key: _sanitize_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_sanitize_json_data(item) for item in data]
    elif isinstance(data, float):
        # Check for invalid float values
        if math.isnan(data) or math.isinf(data):
            return None
        # Check for extremely large values
        if abs(data) > 1e15:
            return None
        return data
    else:
        return data

router = APIRouter(tags=["admin"])

# Cron endpoints (open access for development)
@router.post("/cron/run-selection", response_model=ApiResponse)
async def run_selection_cron():
    """
    Run the daily stock selection pipeline
    Called by Cloud Scheduler or cron
    """
    try:
        logger.info("Starting cron job: run-selection")
        
        summary = selection_engine.run_daily_selection()
        
        response_data = SelectionSummary(**summary)
        
        return ApiResponse(ok=True, data=response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error in selection cron job: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/cron/run-tracker", response_model=ApiResponse)
async def run_tracker_cron():
    """
    Run the position tracker
    Called by Cloud Scheduler or cron
    """
    try:
        logger.info("Starting cron job: run-tracker")
        
        summary = position_tracker.run_position_tracking()
        
        response_data = TrackerSummary(**summary)
        
        return ApiResponse(ok=True, data=response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error in tracker cron job: {e}")
        return ApiResponse(ok=False, error=str(e))

# Admin configuration endpoints
@router.get("/v1/admin/config", response_model=ApiResponse)
async def get_admin_config():
    """Get current runtime configuration"""
    try:
        config_data = firestore_client.get_runtime_config()
        config = dict_to_runtime_config(config_data)
        
        return ApiResponse(ok=True, data=config.model_dump())
        
    except Exception as e:
        logger.error(f"Error getting admin config: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.put("/v1/admin/config", response_model=ApiResponse)
async def update_admin_config(config_update: dict):
    """Update runtime configuration"""
    try:
        # Get current config
        current_config_data = firestore_client.get_runtime_config()
        current_config = dict_to_runtime_config(current_config_data)
        
        # Update with new values
        updated_data = current_config.model_dump()
        updated_data.update(config_update)
        
        # Validate updated config
        try:
            updated_config = RuntimeConfig(**updated_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
        
        # Save to Firestore
        success = firestore_client.set_runtime_config(updated_config.model_dump())
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        # Create audit log
        firestore_client.create_audit_log(
            action="config_updated",
            details={
                "old_config": current_config.model_dump(),
                "new_config": updated_config.model_dump(),
                "changes": config_update
            },
            source="admin_api"
        )
        
        return ApiResponse(ok=True, data=updated_config.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating admin config: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/v1/admin/stats", response_model=ApiResponse)
async def get_admin_stats():
    """Get system statistics and health info"""
    try:
        # Get selection engine stats
        selection_stats = selection_engine.get_selection_stats()
        
        # Get position summary
        position_summary = position_tracker.get_position_summary()
        
        # Get recent signals count
        signals_result = firestore_client.list_signals(status="all", limit=100)
        
        stats = {
            "selection_engine": selection_stats,
            "positions": position_summary,
            "signals": {
                "total_recent": signals_result["total"],
                "open_signals": len([s for s in signals_result["signals"] if s.get("status") == "open"]),
                "placed_signals": len([s for s in signals_result["signals"] if s.get("status") == "placed"])
            },
            "system": {
                "timestamp": firestore_client._add_meta({})["meta"]["created_at"]
            }
        }
        
        return ApiResponse(ok=True, data=stats)
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/v1/admin/test-selection", response_model=ApiResponse)
async def test_selection_pipeline(symbol: str):
    """Test selection pipeline on a single symbol with detailed stage analysis"""
    try:
        from app.services.stocks import stocks_service
        from app.services.indicators import (
            calculate_technical_snapshot, 
            calculate_momentum_score,
            calculate_volume_spike_score,
            calculate_breakout_volatility_score
        )
        from app.services.openai_client import openai_client
        from app.config import settings
        
        # Stage 1: Data Collection
        logger.info(f"Stage 1: Fetching data for {symbol}")
        stock_info = stocks_service.get_stock_info(symbol)
        if not stock_info or stock_info.get('ohlcv') is None:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        ohlcv_data = stock_info['ohlcv']
        fundamentals = stock_info['fundamentals']
        current_price = stock_info.get('current_price')
        
        # Stage 2: Technical Analysis
        logger.info(f"Stage 2: Calculating technical indicators for {symbol}")
        technical = calculate_technical_snapshot(ohlcv_data)
        
        # Stage 3: Component Scoring
        logger.info(f"Stage 3: Calculating component scores for {symbol}")
        momentum_score = calculate_momentum_score(technical) if technical else 0.0
        volume_score = calculate_volume_spike_score(technical, settings.volume_threshold, settings.volume_cap) if technical else 0.0
        breakout_score = calculate_breakout_volatility_score(technical) if technical else 0.0
        
        # Raw score calculation
        raw_score = (
            settings.momentum_weight * momentum_score +
            settings.volume_weight * volume_score +
            settings.breakout_weight * breakout_score
        )
        
        # Stage 4: Fundamental Filtering
        logger.info(f"Stage 4: Checking fundamental sanity for {symbol}")
        fundamental_ok = stocks_service.check_fundamental_sanity(
            fundamentals, 
            settings.min_market_cap_cr, 
            settings.max_pe_ratio
        )
        
        # Stage 5: AI Analysis
        verdict = None
        if fundamental_ok and technical:
            logger.info(f"Stage 5: Getting OpenAI verdict for {symbol}")
            verdict = openai_client.get_stock_verdict(symbol, fundamentals, technical)
        
        # Final blended score
        final_score = 0.0
        if verdict and technical:
            final_score = 0.5 * raw_score + 0.5 * verdict['confidence']
        
        # Detailed result with all stages
        result = {
            "symbol": symbol,
            "stages": {
                "data_collection": {
                    "ohlcv_days": len(ohlcv_data) if ohlcv_data is not None else 0,
                    "current_price": current_price,
                    "data_quality": "good" if len(ohlcv_data) >= 30 else "insufficient"
                },
                "technical_analysis": {
                    "indicators_calculated": len(technical) if technical else 0,
                    "technical_data": technical,
                    "status": "success" if technical else "failed"
                },
                "component_scoring": {
                    "momentum_score": momentum_score,
                    "volume_score": volume_score,
                    "breakout_score": breakout_score,
                    "raw_score": raw_score,
                    "weights": {
                        "momentum": settings.momentum_weight,
                        "volume": settings.volume_weight,
                        "breakout": settings.breakout_weight
                    }
                },
                "fundamental_filtering": {
                    "fundamental_ok": fundamental_ok,
                    "fundamentals": fundamentals,
                    "filters": {
                        "min_market_cap_cr": settings.min_market_cap_cr,
                        "max_pe_ratio": settings.max_pe_ratio
                    }
                },
                "ai_analysis": {
                    "verdict": verdict,
                    "status": "success" if verdict else "failed"
                },
                "final_scoring": {
                    "final_score": final_score,
                    "meets_threshold": final_score >= settings.min_signal_score,
                    "threshold": settings.min_signal_score
                }
            },
            "summary": {
                "fundamental_ok": fundamental_ok,
                "technical_ok": bool(technical),
                "ai_verdict_ok": bool(verdict),
                "overall_success": fundamental_ok and bool(technical) and bool(verdict),
                "data_points": len(ohlcv_data) if ohlcv_data is not None else 0
            }
        }
        
        return ApiResponse(ok=True, data=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing selection for {symbol}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/v1/admin/test-enhanced-selection", response_model=ApiResponse)
async def test_enhanced_selection_pipeline(symbol: str):
    """Test enhanced selection pipeline on a single symbol with detailed analysis"""
    try:
        logger.info(f"🚀 ENHANCED SELECTION TEST STARTED for {symbol}")
        
        from app.services.enhanced_selection import enhanced_selection_engine
        from app.services.stocks import stocks_service
        from app.services.enhanced_scoring import enhanced_scoring
        
        # Stage 1: Enhanced Data Collection
        logger.info(f"📊 Stage 1: Fetching enhanced data for {symbol}")
        stock_info = stocks_service.get_enhanced_stock_info(symbol)
        if not stock_info or stock_info.get('ohlcv') is None:
            logger.error(f"❌ No data found for symbol {symbol}")
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        ohlcv_data = stock_info['ohlcv']
        fundamentals = stock_info['fundamentals']
        enhanced_technical = stock_info.get('enhanced_technical', {})
        enhanced_fundamentals = stock_info.get('enhanced_fundamentals', {})
        fundamental_score_data = stock_info.get('fundamental_score', {})
        current_price = stock_info.get('current_price')
        
        logger.info(f"✅ Data collected - OHLCV: {len(ohlcv_data)} days, Technical indicators: {len(enhanced_technical)}")
        logger.info(f"📈 Current price: {current_price}, Enhanced technical available: {bool(enhanced_technical)}")
        logger.info(f"💰 Enhanced fundamentals available: {bool(enhanced_fundamentals)}, Fundamental score: {fundamental_score_data.get('final_score', 'N/A')}")
        
        # Stage 2: Enhanced Technical Analysis
        logger.info(f"🔍 Stage 2: Enhanced technical analysis for {symbol}")
        logger.info(f"📋 Technical indicators breakdown:")
        logger.info(f"   - Basic: {len([k for k in enhanced_technical.keys() if k in ['sma_20', 'sma_50', 'rsi_14', 'atr_14', 'current_price']])}")
        logger.info(f"   - Momentum: {len([k for k in enhanced_technical.keys() if k in ['macd', 'macd_signal', 'macd_histogram', 'stoch_rsi_k', 'williams_r', 'roc_5', 'roc_10', 'roc_20']])}")
        logger.info(f"   - Volume: {len([k for k in enhanced_technical.keys() if k in ['vwap', 'vwap_upper', 'vwap_lower', 'obv', 'ad_line']])}")
        logger.info(f"   - Multi-timeframe: {len([k for k in enhanced_technical.keys() if k.endswith('_trend')])}")
        
        technical_indicators = {
            "basic_indicators": {
                "sma_20": enhanced_technical.get("sma_20"),
                "sma_50": enhanced_technical.get("sma_50"),
                "rsi_14": enhanced_technical.get("rsi_14"),
                "atr_14": enhanced_technical.get("atr_14")
            },
            "momentum_indicators": {
                "macd": enhanced_technical.get("macd"),
                "macd_signal": enhanced_technical.get("macd_signal"),
                "macd_histogram": enhanced_technical.get("macd_histogram"),
                "stoch_rsi_k": enhanced_technical.get("stoch_rsi_k"),
                "stoch_rsi_d": enhanced_technical.get("stoch_rsi_d"),
                "williams_r": enhanced_technical.get("williams_r"),
                "roc_5": enhanced_technical.get("roc_5"),
                "roc_10": enhanced_technical.get("roc_10"),
                "roc_20": enhanced_technical.get("roc_20")
            },
            "volume_indicators": {
                "vwap": enhanced_technical.get("vwap"),
                "vwap_upper": enhanced_technical.get("vwap_upper"),
                "vwap_lower": enhanced_technical.get("vwap_lower"),
                "obv": enhanced_technical.get("obv"),
                "ad_line": enhanced_technical.get("ad_line")
            },
            "divergence_signals": enhanced_technical.get("rsi_divergence", {}),
            "multi_timeframe": {
                "1m_trend": enhanced_technical.get("1m_trend"),
                "5m_trend": enhanced_technical.get("5m_trend"),
                "15m_trend": enhanced_technical.get("15m_trend"),
                "1d_trend": enhanced_technical.get("1d_trend"),
                "1wk_trend": enhanced_technical.get("1wk_trend")
            }
        }
        
        # Stage 3: Enhanced Technical Scoring
        logger.info(f"🎯 Stage 3: Calculating enhanced technical scores for {symbol}")
        enhanced_score_data = enhanced_scoring.calculate_enhanced_score(enhanced_technical)
        logger.info(f"📊 Technical score: {enhanced_score_data.get('final_score', 0):.3f}, Signal strength: {enhanced_score_data.get('signal_strength', 'unknown')}")
        logger.info(f"🎚️ Technical component scores: {enhanced_score_data.get('component_scores', {})}")
        
        # Stage 3.5: Enhanced Fundamental Analysis
        logger.info(f"💰 Stage 3.5: Enhanced fundamental analysis for {symbol}")
        if enhanced_fundamentals:
            logger.info(f"📋 Fundamental metrics breakdown:")
            quality_metrics = enhanced_fundamentals.get('quality_metrics', {})
            growth_metrics = enhanced_fundamentals.get('growth_metrics', {})
            value_metrics = enhanced_fundamentals.get('value_metrics', {})
            momentum_metrics = enhanced_fundamentals.get('momentum_metrics', {})
            
            logger.info(f"   - Quality: {len([k for k, v in quality_metrics.items() if v is not None])} metrics")
            logger.info(f"   - Growth: {len([k for k, v in growth_metrics.items() if v is not None])} metrics")
            logger.info(f"   - Value: {len([k for k, v in value_metrics.items() if v is not None])} metrics")
            logger.info(f"   - Momentum: {len([k for k, v in momentum_metrics.items() if v is not None])} metrics")
            
            logger.info(f"📊 Fundamental score: {fundamental_score_data.get('final_score', 0):.3f}, Strength: {fundamental_score_data.get('fundamental_strength', 'unknown')}")
            logger.info(f"🎚️ Fundamental component scores: {fundamental_score_data.get('component_scores', {})}")
        else:
            logger.warning(f"⚠️ No enhanced fundamental data available for {symbol}")
        
        # Stage 4: Combined Scoring
        logger.info(f"🎯 Stage 4: Calculating combined scores for {symbol}")
        technical_score = enhanced_score_data.get("final_score", 0.0)
        fundamental_score = fundamental_score_data.get("final_score", 0.5) if fundamental_score_data else 0.5
        
        # Combined score: 60% technical + 40% fundamental
        combined_score = (0.6 * technical_score + 0.4 * fundamental_score)
        
        logger.info(f"📊 Combined score: {combined_score:.3f} (Technical: {technical_score:.3f}, Fundamental: {fundamental_score:.3f})")
        logger.info(f"⚖️ Scoring weights - Technical: 60%, Fundamental: 40%")
        
        # Stage 5: Enhanced Filtering
        logger.info(f"🔍 Stage 5: Checking enhanced filters for {symbol}")
        fundamental_ok = stocks_service.check_fundamental_sanity(
            fundamentals, 
            settings.min_market_cap_cr, 
            settings.max_pe_ratio
        )
        
        passes_enhanced_filters = enhanced_selection_engine._passes_enhanced_filters(
            enhanced_score_data, enhanced_technical
        )
        
        logger.info(f"✅ Fundamental check: {fundamental_ok}, Enhanced filters: {passes_enhanced_filters}")
        logger.info(f"🎛️ Filter criteria - Min confidence: {enhanced_selection_engine.min_confidence}, Require divergence: {enhanced_selection_engine.require_divergence}, Require MTF alignment: {enhanced_selection_engine.require_mtf_alignment}")
        
        # Stage 6: Multi-Stage AI Analysis
        multi_stage_analysis = None
        verdict = None
        if fundamental_ok and enhanced_technical:
            logger.info(f"🤖 Stage 6: Running multi-stage AI analysis for {symbol}")
            # Count total metrics in enhanced fundamentals
            total_fundamental_metrics = 0
            if enhanced_fundamentals:
                for category in ['quality_metrics', 'growth_metrics', 'value_metrics', 'momentum_metrics']:
                    category_data = enhanced_fundamentals.get(category, {})
                    total_fundamental_metrics += len([k for k, v in category_data.items() if v is not None])
            
            logger.info(f"📤 Sending to Multi-Stage AI - Technical keys: {len(enhanced_technical)} indicators, Fundamental keys: {total_fundamental_metrics} metrics")
            logger.info(f"🔍 Enhanced fundamentals structure: {list(enhanced_fundamentals.keys()) if enhanced_fundamentals else 'None'}")
            
            from app.services.multi_stage_prompting import multi_stage_prompting_service
            multi_stage_analysis = multi_stage_prompting_service.analyze_stock(
                symbol, fundamentals, enhanced_technical, enhanced_fundamentals
            )
            
            if multi_stage_analysis and not multi_stage_analysis.get("error"):
                verdict = multi_stage_analysis.get("final_recommendation", {})
                logger.info(f"📥 Multi-Stage AI Response - Action: {verdict.get('action', 'unknown')}, Confidence: {verdict.get('confidence', 0):.2f}")
                logger.info(f"🎯 Selected Module: {multi_stage_analysis.get('analysis_stages', {}).get('stage2_module', {}).get('selected_module', 'unknown')}")
                logger.info(f"⚠️ Risk Level: {multi_stage_analysis.get('risk_summary', {}).get('risk_level', 'unknown')}")
            else:
                logger.warning(f"⚠️ Multi-stage analysis failed for {symbol}: {multi_stage_analysis.get('error', 'Unknown error') if multi_stage_analysis else 'No response'}")
        else:
            logger.warning(f"⚠️ Skipping multi-stage analysis - Fundamental OK: {fundamental_ok}, Technical available: {bool(enhanced_technical)}")
        
        # Final blended score
        final_score = 0.0
        if verdict and enhanced_technical:
            llm_confidence = verdict['confidence']
            technical_confidence = enhanced_score_data.get("confidence", 0.5)
            fundamental_confidence = fundamental_score_data.get("confidence", 0.5) if fundamental_score_data else 0.5
            
            # Enhanced blending: 50% combined score, 30% LLM, 20% confidence blend
            confidence_blend = (technical_confidence + fundamental_confidence) / 2
            final_score = (0.5 * combined_score + 0.3 * llm_confidence + 0.2 * confidence_blend)
            
            logger.info(f"🎯 Final blended score: {final_score:.3f} (Combined: {combined_score:.3f}, LLM: {llm_confidence:.3f}, Confidence: {confidence_blend:.3f})")
            logger.info(f"⚖️ Final blending weights - Combined: 50%, LLM: 30%, Confidence: 20%")
        else:
            logger.warning(f"⚠️ Cannot calculate final score - Verdict: {bool(verdict)}, Technical: {bool(enhanced_technical)}")
        
        # Detailed result with all stages
        result = {
            "symbol": symbol,
            "stages": {
                "data_collection": {
                    "ohlcv_days": len(ohlcv_data) if ohlcv_data is not None else 0,
                    "current_price": current_price,
                    "enhanced_technical_available": bool(enhanced_technical),
                    "enhanced_fundamentals_available": bool(enhanced_fundamentals),
                    "data_quality": "good" if len(ohlcv_data) >= 30 else "insufficient"
                },
                "enhanced_technical_analysis": {
                    "indicators_available": len([k for k, v in enhanced_technical.items() if v is not None]),
                    "technical_indicators": technical_indicators,
                    "status": "success" if enhanced_technical else "failed"
                },
                "enhanced_technical_scoring": {
                    "technical_score": enhanced_score_data.get("final_score", 0.0),
                    "signal_strength": enhanced_score_data.get("signal_strength", "neutral"),
                    "confidence": enhanced_score_data.get("confidence", 0.0),
                    "component_scores": enhanced_score_data.get("component_scores", {}),
                    "scoring_weights": enhanced_scoring.weights
                },
                "enhanced_fundamental_analysis": {
                    "fundamental_score": fundamental_score_data.get("final_score", 0.0),
                    "fundamental_strength": fundamental_score_data.get("fundamental_strength", "average"),
                    "fundamental_confidence": fundamental_score_data.get("confidence", 0.0),
                    "component_scores": fundamental_score_data.get("component_scores", {}),
                    "quality_metrics": enhanced_fundamentals.get("quality_metrics", {}),
                    "growth_metrics": enhanced_fundamentals.get("growth_metrics", {}),
                    "value_metrics": enhanced_fundamentals.get("value_metrics", {}),
                    "momentum_metrics": enhanced_fundamentals.get("momentum_metrics", {}),
                    "status": "success" if enhanced_fundamentals else "failed"
                },
                "combined_scoring": {
                    "technical_score": technical_score,
                    "fundamental_score": fundamental_score,
                    "combined_score": combined_score,
                    "scoring_weights": {"technical": 0.6, "fundamental": 0.4}
                },
                "enhanced_filtering": {
                    "fundamental_ok": fundamental_ok,
                    "passes_enhanced_filters": passes_enhanced_filters,
                    "min_confidence": enhanced_selection_engine.min_confidence,
                    "require_divergence": enhanced_selection_engine.require_divergence,
                    "require_mtf_alignment": enhanced_selection_engine.require_mtf_alignment
                },
                "multi_stage_ai_analysis": {
                    "verdict": verdict,
                    "multi_stage_analysis": multi_stage_analysis,
                    "status": "success" if verdict else "failed",
                    "selected_module": multi_stage_analysis.get("analysis_stages", {}).get("stage2_module", {}).get("selected_module") if multi_stage_analysis else None,
                    "risk_level": multi_stage_analysis.get("risk_summary", {}).get("risk_level") if multi_stage_analysis else None
                },
                "final_scoring": {
                    "final_score": final_score,
                    "meets_threshold": final_score >= settings.min_signal_score,
                    "threshold": settings.min_signal_score,
                    "blending_method": "enhanced_50_30_20",
                    "blending_weights": {"combined_score": 0.5, "llm_confidence": 0.3, "confidence_blend": 0.2}
                }
            },
            "summary": {
                "fundamental_ok": fundamental_ok,
                "enhanced_technical_ok": bool(enhanced_technical),
                "enhanced_fundamentals_ok": bool(enhanced_fundamentals),
                "enhanced_filters_ok": passes_enhanced_filters,
                "multi_stage_ai_ok": bool(multi_stage_analysis and not multi_stage_analysis.get("error")),
                "ai_verdict_ok": bool(verdict),
                "overall_success": fundamental_ok and bool(enhanced_technical) and passes_enhanced_filters and bool(verdict),
                "data_points": len(ohlcv_data) if ohlcv_data is not None else 0,
                "enhanced_analysis": True,
                "scoring_method": "enhanced_technical_fundamental_multi_stage_v4"
            }
        }
        
        logger.info(f"✅ ENHANCED SELECTION TEST COMPLETED for {symbol}")
        logger.info(f"📊 Final result - Overall success: {result['summary']['overall_success']}, Final score: {final_score:.3f}, Meets threshold: {final_score >= settings.min_signal_score}")
        
        # Sanitize the result to ensure JSON compliance
        sanitized_result = _sanitize_json_data(result)
        
        return ApiResponse(ok=True, data=sanitized_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error testing enhanced selection for {symbol}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/v1/admin/run-enhanced-selection", response_model=ApiResponse)
async def run_enhanced_selection():
    """Run the enhanced selection pipeline"""
    try:
        from app.services.enhanced_selection import enhanced_selection_engine
        
        logger.info("Starting enhanced selection pipeline")
        
        # Run enhanced selection
        summary = enhanced_selection_engine.run_enhanced_selection()
        
        return ApiResponse(ok=True, data=summary)
        
    except Exception as e:
        logger.error(f"Error running enhanced selection: {e}")
        return ApiResponse(ok=False, error=str(e))

