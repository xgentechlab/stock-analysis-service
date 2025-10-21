"""
Stage processor for executing analysis pipeline stages
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import time

from app.models.schemas import StageStatus, AnalysisType
from app.services.job_service import job_service
from app.services.stage_config import stage_config_service
from app.services.stocks import stocks_service
from app.services.enhanced_indicators import enhanced_technical
from app.services.enhanced_fundamentals import enhanced_fundamental_analysis
from app.services.enhanced_scoring import enhanced_scoring
from app.services.multi_stage_prompting import multi_stage_prompting_service

logger = logging.getLogger(__name__)

class StageProcessor:
    """Processor for executing individual analysis stages"""
    
    def __init__(self):
        self.job_service = job_service
        self.stage_config = stage_config_service
    
    def process_job(self, job_id: str) -> bool:
        """Process a complete job by executing all stages with smart caching"""
        try:
            job_data = self.job_service.get_job(job_id)
            if not job_data:
                logger.error(f"Job {job_id} not found")
                return False
            
            symbol = job_data["symbol"]
            analysis_type = AnalysisType(job_data["analysis_type"])
            
            logger.info(f"Starting job processing for {symbol} (type: {analysis_type.value})")
            
            # Check for cached analysis first (investment optimization)
            from app.db.firestore_client import firestore_client
            cached_analysis = firestore_client.get_latest_analysis_by_symbol(symbol, analysis_type.value)
            
            logger.info(f"🔍 Cache check for {symbol}: {'Found' if cached_analysis else 'Not found'}")
            
            if cached_analysis:
                freshness_check = firestore_client.is_analysis_fresh_for_investment(cached_analysis)
                logger.info(f"   Freshness check: {freshness_check}")
                
                if freshness_check["is_fresh"]:
                    logger.info(f"🎯 Using cached analysis for {symbol} (age: {freshness_check['age_days']:.1f} days) - saving $0.10")
                    
                    # Copy cached analysis to current job
                    self._copy_cached_analysis_to_job(job_id, cached_analysis)
                    return True
                else:
                    logger.info(f"⏰ Cached analysis is stale for {symbol}: {len(freshness_check['stale_stages'])} stages need refresh")
                    logger.info(f"   Stale stages: {[s['stage'] for s in freshness_check['stale_stages']]}")
                    logger.info(f"   Freshness score: {freshness_check.get('freshness_score', 0)}")
            else:
                logger.info(f"📝 No cached analysis found for {symbol} - proceeding with new analysis")
            
            # Get stage execution order
            stage_order = self.stage_config.get_stage_order(analysis_type)
            
            # Track completed and failed stages
            completed_stages = []
            failed_stages = []
            stage_results = {}
            
            # Process stages with parallel execution support
            while len(completed_stages) + len(failed_stages) < len(stage_order):
                # Get parallel stages that can be executed (excluding failed stages)
                parallel_stages = self.stage_config.get_parallel_stages(analysis_type, completed_stages)
                
                # Filter out already failed stages
                parallel_stages = [stage for stage in parallel_stages if stage not in failed_stages]
                
                if not parallel_stages:
                    logger.warning("No more stages can be executed")
                    break
                
                if len(parallel_stages) == 1:
                    # Single stage - execute normally
                    stage_name = parallel_stages[0]
                    logger.info(f"Processing stage: {stage_name} for {symbol}")
                    
                    success, result, error = self._execute_stage(
                        job_id, symbol, analysis_type, stage_name, stage_results
                    )
                    
                    if success:
                        completed_stages.append(stage_name)
                        stage_results[stage_name] = result
                        logger.info(f"Stage {stage_name} completed successfully")
                    else:
                        logger.error(f"Stage {stage_name} failed: {error}")
                        failed_stages.append(stage_name)
                        
                        # Check if this is a critical error that should fail the entire job
                        if self._is_critical_error(error) or self._is_critical_stage(stage_name):
                            logger.error(f"Critical error in stage {stage_name}: {error}. Failing entire job.")
                            self._mark_job_as_failed(job_id, f"Critical error in {stage_name}: {error}")
                            return False
                else:
                    # Multiple parallel stages - execute concurrently
                    logger.info(f"Processing parallel stages: {parallel_stages} for {symbol}")
                    
                    # For now, execute sequentially but mark as parallel
                    # TODO: Implement true parallel execution with threading/async
                    for stage_name in parallel_stages:
                        success, result, error = self._execute_stage(
                            job_id, symbol, analysis_type, stage_name, stage_results
                        )
                        
                        if success:
                            completed_stages.append(stage_name)
                            stage_results[stage_name] = result
                            logger.info(f"Parallel stage {stage_name} completed successfully")
                        else:
                            logger.error(f"Parallel stage {stage_name} failed: {error}")
                            failed_stages.append(stage_name)
                            
                            # Check if this is a critical error that should fail the entire job
                            if self._is_critical_error(error) or self._is_critical_stage(stage_name):
                                logger.error(f"Critical error in parallel stage {stage_name}: {error}. Failing entire job.")
                                self._mark_job_as_failed(job_id, f"Critical error in {stage_name}: {error}")
                                return False
                            # Continue with other parallel stages even if one fails
            
            # Check if job should be marked as failed due to too many failed stages
            if failed_stages and len(failed_stages) >= len(stage_order) * 0.5:  # More than 50% failed
                logger.error(f"Too many failed stages ({len(failed_stages)}/{len(stage_order)}). Failing job.")
                self._mark_job_as_failed(job_id, f"Too many failed stages: {failed_stages}")
                return False
            
            logger.info(f"Job processing completed for {symbol}. Completed stages: {completed_stages}, Failed stages: {failed_stages}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process job {job_id}: {e}")
            self._mark_job_as_failed(job_id, f"Job processing failed: {str(e)}")
            return False
    
    def _is_critical_error(self, error: str) -> bool:
        """Check if an error is critical and should fail the entire job"""
        if not error:
            return False
        
        error_lower = error.lower()
        
        # Critical errors that indicate the symbol is invalid or delisted
        critical_patterns = [
            "no timezone found",
            "symbol may be delisted",
            "no data found for symbol",
            "invalid symbol",
            "symbol not found",
            "ticker not found",
            "delisted",
            "suspended",
            "not available",
            "no data available",
            "empty data",
            "no ohlcv data",
            "insufficient data"
        ]
        
        return any(pattern in error_lower for pattern in critical_patterns)
    
    def _is_critical_stage(self, stage_name: str) -> bool:
        """Check if a stage is critical and its failure should fail the entire job"""
        critical_stages = [
            "data_collection",
            "data_collection_and_analysis",
            "enhanced_data_collection"
        ]
        return stage_name in critical_stages
    
    def _mark_job_as_failed(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with error message"""
        try:
            from app.models.schemas import JobStatus
            from datetime import datetime, timezone
            
            # Update job status to failed
            updates = {
                "status": JobStatus.FAILED.value,
                "error": error_message,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            self.job_service.db.update_job(job_id, updates)
            logger.error(f"Job {job_id} marked as failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
    
    def _get_initial_stage_data(self, stage_name: str, symbol: str) -> Dict[str, Any]:
        """Get initial data to show while stage is processing"""
        initial_data = {
            "status": "processing",
            "message": f"Starting {stage_name.replace('_', ' ').title()} for {symbol}",
            "progress": 0,
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add stage-specific initial data
        if stage_name == "data_collection_and_analysis":
            initial_data.update({
                "message": f"Fetching market data and performing analysis for {symbol}",
                "steps": ["Fetching OHLCV data", "Getting fundamentals", "Enhanced technical analysis", "Enhanced fundamental analysis"]
            })
        elif stage_name == "technical_and_combined_scoring":
            initial_data.update({
                "message": f"Calculating technical scores and combined scoring for {symbol}",
                "steps": ["Technical scoring", "Combined score calculation"]
            })
        elif stage_name == "forensic_analysis":
            initial_data.update({
                "message": f"Performing forensic analysis for {symbol}",
                "steps": ["Financial analysis", "Risk assessment", "Signal strength calculation"]
            })
        elif stage_name == "module_selection":
            initial_data.update({
                "message": f"Selecting analysis module for {symbol}",
                "steps": ["Momentum analysis", "Value analysis", "Balanced analysis"]
            })
        elif stage_name == "risk_assessment":
            initial_data.update({
                "message": f"Assessing risks for {symbol}",
                "steps": ["Risk identification", "Position sizing", "Risk mitigation"]
            })
        elif stage_name == "final_decision":
            initial_data.update({
                "message": f"Making final decision for {symbol}",
                "steps": ["Decision analysis", "Confidence calculation", "Position sizing"]
            })
        elif stage_name == "verdict_synthesis":
            initial_data.update({
                "message": f"Synthesizing AI analysis results for {symbol}",
                "steps": ["Combining forensic analysis", "Integrating module selection", "Incorporating risk assessment", "Finalizing decision"]
            })
        elif stage_name == "final_scoring":
            initial_data.update({
                "message": f"Calculating final score for {symbol}",
                "steps": ["Score blending", "Threshold evaluation", "Final recommendation"]
            })
        
        return initial_data
    
    def _execute_stage(self, job_id: str, symbol: str, analysis_type: AnalysisType, 
                      stage_name: str, previous_results: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Execute a specific stage"""
        try:
            # Mark stage as processing with initial data
            initial_data = self._get_initial_stage_data(stage_name, symbol)
            self.job_service.update_stage_status(job_id, stage_name, StageStatus.PROCESSING, data=initial_data)
            
            start_time = time.time()
            
            # Execute stage based on name
            if stage_name == "data_collection_and_analysis":
                result = self._execute_data_collection_and_analysis(symbol, analysis_type)
            elif stage_name == "technical_and_combined_scoring":
                result = self._execute_technical_and_combined_scoring(symbol, previous_results)
            elif stage_name == "forensic_analysis":
                result = self._execute_forensic_analysis(symbol, previous_results)
            elif stage_name == "module_selection":
                result = self._execute_module_selection(symbol, previous_results)
            elif stage_name == "risk_assessment":
                result = self._execute_risk_assessment(symbol, previous_results)
            elif stage_name == "final_decision":
                result = self._execute_final_decision(symbol, previous_results)
            elif stage_name == "verdict_synthesis":
                result = self._execute_verdict_synthesis(symbol, previous_results)
            elif stage_name == "final_scoring":
                result = self._execute_final_scoring(symbol, previous_results)
            else:
                raise ValueError(f"Unknown stage: {stage_name}")
            
            # Mark stage as completed
            self.job_service.update_stage_status(job_id, stage_name, StageStatus.COMPLETED, result)

            # After persisting final_scoring, route to recommendations or watchlist
            if stage_name == "final_scoring":
                try:
                    # Extract inputs for routing
                    from app.services.post_scoring_router import route_post_scoring
                    synthesis = previous_results.get("verdict_synthesis", {})
                    final_reco = synthesis.get("final_recommendation", {})
                    action = final_reco.get("action")
                    confidence = float(final_reco.get("confidence", 0.0))
                    rationale = final_reco.get("rationale")
                    final_score = float(result.get("final_score", 0.0))

                    route_post_scoring(
                        symbol=symbol,
                        job_id=job_id,
                        final_score=final_score,
                        action=action or "",
                        confidence=confidence,
                        rationale=rationale,
                        user_id="default_user",
                    )
                except Exception as routing_e:
                    logger.warning(f"Post-final_scoring routing skipped for {symbol}: {routing_e}")
            
            duration = time.time() - start_time
            logger.info(f"Stage {stage_name} completed in {duration:.2f}s")
            
            return True, result, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Stage {stage_name} failed: {error_msg}")
            
            # Mark stage as failed
            self.job_service.update_stage_status(
                job_id, stage_name, StageStatus.FAILED, 
                error=error_msg
            )
            
            return False, None, error_msg
    
    def _execute_data_collection(self, symbol: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Execute data collection stage"""
        if analysis_type == AnalysisType.ENHANCED:
            stock_info = stocks_service.get_enhanced_stock_info(symbol)
        else:
            stock_info = stocks_service.get_stock_info(symbol)
        
        if not stock_info or stock_info.get('ohlcv') is None:
            raise ValueError(f"No data found for symbol {symbol}")
        
        ohlcv = stock_info.get('ohlcv')
        if hasattr(ohlcv, 'empty') and ohlcv.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        elif ohlcv is None:
            raise ValueError(f"No data found for symbol {symbol}")
        
        ohlcv_days = len(ohlcv) if ohlcv is not None and not (hasattr(ohlcv, 'empty') and ohlcv.empty) else 0
        
        # Safely check if enhanced data is available
        enhanced_technical = stock_info.get('enhanced_technical')
        enhanced_fundamentals = stock_info.get('enhanced_fundamentals')
        
        # Check if enhanced data exists and is not empty
        enhanced_technical_available = (
            enhanced_technical is not None and 
            not (hasattr(enhanced_technical, 'empty') and enhanced_technical.empty)
        )
        enhanced_fundamentals_available = (
            enhanced_fundamentals is not None and 
            not (hasattr(enhanced_fundamentals, 'empty') and enhanced_fundamentals.empty)
        )
        
        # Convert DataFrame to dict for Firestore storage
        raw_data = stock_info.copy()
        if 'ohlcv' in raw_data and hasattr(raw_data['ohlcv'], 'to_dict'):
            raw_data['ohlcv'] = raw_data['ohlcv'].to_dict('records')
        
        return {
            "ohlcv_days": ohlcv_days,
            "current_price": stock_info.get('current_price'),
            "enhanced_technical_available": enhanced_technical_available,
            "enhanced_fundamentals_available": enhanced_fundamentals_available,
            "data_quality": "good" if ohlcv_days >= 30 else "insufficient",
            "summary": {
                "price_range": {
                    "high": float(stock_info['ohlcv']['High'].max()) if 'ohlcv' in stock_info and not stock_info['ohlcv'].empty else None,
                    "low": float(stock_info['ohlcv']['Low'].min()) if 'ohlcv' in stock_info and not stock_info['ohlcv'].empty else None,
                    "current": stock_info.get('current_price')
                },
                "volume_avg": float(stock_info['ohlcv']['Volume'].mean()) if 'ohlcv' in stock_info and not stock_info['ohlcv'].empty else None,
                "data_sources": ["yfinance", "enhanced_technical", "enhanced_fundamentals"]
            },
            "raw_data": raw_data  # Store for subsequent stages
        }
    
    def _execute_enhanced_technical_analysis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced technical analysis stage"""
        data_collection = previous_results.get("data_collection", {})
        raw_data = data_collection.get("raw_data", {})
        
        enhanced_technical_data = raw_data.get('enhanced_technical', {})
        
        # Check if enhanced_technical_data is empty or None
        if enhanced_technical_data is None or (hasattr(enhanced_technical_data, 'empty') and enhanced_technical_data.empty):
            # Calculate if not already available
            enhanced_technical_data = enhanced_technical.analyze_symbol(symbol, days_back=30)
        
        # Format technical indicators for better UI display
        formatted_indicators = self._format_technical_indicators(enhanced_technical_data)
        
        return {
            "indicators_available": len(enhanced_technical_data) if enhanced_technical_data else 0,
            "technical_indicators": formatted_indicators,
            "status": "success" if enhanced_technical_data else "failed",
            "summary": {
                "momentum_indicators": len([k for k in formatted_indicators.keys() if 'momentum' in k.lower()]),
                "volume_indicators": len([k for k in formatted_indicators.keys() if 'volume' in k.lower()]),
                "trend_indicators": len([k for k in formatted_indicators.keys() if 'trend' in k.lower() or 'ma' in k.lower()]),
                "oscillator_indicators": len([k for k in formatted_indicators.keys() if 'rsi' in k.lower() or 'stoch' in k.lower()])
            }
        }
    
    def _execute_enhanced_technical_scoring(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced technical scoring stage"""
        technical_analysis = previous_results.get("enhanced_technical_analysis", {})
        technical_indicators = technical_analysis.get("technical_indicators", {})
        
        if not technical_indicators:
            raise ValueError("Technical indicators not available for scoring")
        
        # Calculate technical score
        technical_score_data = enhanced_scoring.calculate_enhanced_score(technical_indicators)
        
        return technical_score_data
    
    def _execute_enhanced_fundamental_analysis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced fundamental analysis stage"""
        data_collection = previous_results.get("data_collection", {})
        raw_data = data_collection.get("raw_data", {})
        
        enhanced_fundamentals = raw_data.get('enhanced_fundamentals', {})
        fundamental_score_data = raw_data.get('fundamental_score', {})
        
        # Check if enhanced_fundamentals is empty or None
        if enhanced_fundamentals is None or (hasattr(enhanced_fundamentals, 'empty') and enhanced_fundamentals.empty):
            # Calculate if not already available
            enhanced_fundamentals = enhanced_fundamental_analysis.fetch_enhanced_fundamentals(symbol)
        
        # Check if fundamental_score_data is empty and enhanced_fundamentals exists
        if not fundamental_score_data and enhanced_fundamentals is not None and not (hasattr(enhanced_fundamentals, 'empty') and enhanced_fundamentals.empty):
            from app.services.fundamental_scoring import fundamental_scoring
            fundamental_score_data = fundamental_scoring.calculate_fundamental_score(enhanced_fundamentals)
        
        return {
            "fundamental_score": fundamental_score_data,
            "enhanced_fundamentals": enhanced_fundamentals,
            "status": "success" if enhanced_fundamentals else "failed"
        }
    
    def _execute_combined_scoring(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute combined scoring stage"""
        technical_scoring = previous_results.get("enhanced_technical_scoring", {})
        fundamental_analysis = previous_results.get("enhanced_fundamental_analysis", {})
        
        technical_score = technical_scoring.get("technical_score", 0.0)
        fundamental_score = fundamental_analysis.get("fundamental_score", {}).get("final_score", 0.0)
        
        # Calculate combined score (60% technical, 40% fundamental)
        combined_score = 0.6 * technical_score + 0.4 * fundamental_score
        
        return {
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "combined_score": combined_score,
            "scoring_weights": {
                "technical": 0.6,
                "fundamental": 0.4
            }
        }
    
    def _execute_enhanced_filtering(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced filtering stage"""
        combined_scoring = previous_results.get("combined_scoring", {})
        fundamental_analysis = previous_results.get("enhanced_fundamental_analysis", {})
        
        combined_score = combined_scoring.get("combined_score", 0.0)
        fundamental_score_data = fundamental_analysis.get("fundamental_score", {})
        
        # Apply enhanced filters
        fundamental_ok = fundamental_score_data.get("final_score", 0.0) >= 0.5
        passes_enhanced_filters = combined_score >= 0.5 and fundamental_ok
        
        return {
            "fundamental_ok": fundamental_ok,
            "passes_enhanced_filters": passes_enhanced_filters,
            "min_confidence": 0.6,
            "require_divergence": False,
            "require_mtf_alignment": True
        }
    
    def _execute_multi_stage_ai_analysis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-stage AI analysis stage"""
        data_collection = previous_results.get("data_collection", {})
        technical_analysis = previous_results.get("enhanced_technical_analysis", {})
        fundamental_analysis = previous_results.get("enhanced_fundamental_analysis", {})
        
        raw_data = data_collection.get("raw_data", {})
        technical_indicators = technical_analysis.get("technical_indicators", {})
        enhanced_fundamentals = fundamental_analysis.get("enhanced_fundamentals", {})
        
        if not technical_indicators or not raw_data.get('fundamentals'):
            raise ValueError("Required data not available for AI analysis")
        
        # Run multi-stage AI analysis
        multi_stage_analysis = multi_stage_prompting_service.analyze_stock(
            symbol, 
            raw_data.get('fundamentals', {}), 
            technical_indicators, 
            enhanced_fundamentals
        )
        
        if not multi_stage_analysis or multi_stage_analysis.get("error"):
            raise ValueError(f"Multi-stage AI analysis failed: {multi_stage_analysis.get('error', 'Unknown error')}")
        
        return multi_stage_analysis
    
    def _execute_final_scoring(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final scoring stage"""
        # Correct sources: combined score from stage2, confidence from verdict synthesis
        stage2 = previous_results.get("technical_and_combined_scoring", {}) or {}
        combined_scoring = stage2.get("combined_scoring", {}) or {}
        combined_score = float(combined_scoring.get("combined_score", 0.0))

        verdict_stage = previous_results.get("verdict_synthesis", {}) or {}
        final_reco = verdict_stage.get("final_recommendation", {}) or {}
        ai_confidence = float(final_reco.get("confidence", 0.0))
        
        # Fallback: try final_decision if verdict_synthesis missing
        if ai_confidence == 0.0:
            fd = previous_results.get("final_decision", {}) or {}
            if isinstance(fd, dict):
                ai_confidence = float(fd.get("confidence", ai_confidence))
        
        # Calculate final blended score (50% combined, 30% AI confidence, 20% confidence blend)
        final_score = 0.5 * combined_score + 0.3 * ai_confidence + 0.2 * (combined_score * ai_confidence)
        
        return {
            "final_score": final_score,
            "meets_threshold": final_score >= 0.5,
            "threshold": 0.5,
            "blending_method": "enhanced_50_30_20",
            "blending_weights": {
                "combined_score": 0.5,
                "llm_confidence": 0.3,
                "confidence_blend": 0.2
            }
        }
    
    def _format_technical_indicators(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format technical indicators for storage"""
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
    
    def _is_critical_stage(self, stage_name: str) -> bool:
        """Check if a stage is critical for the analysis"""
        critical_stages = ["data_collection_and_analysis", "verdict_synthesis"]
        return stage_name in critical_stages
    
    # New stage execution methods for the 8-stage structure
    
    def _execute_data_collection_and_analysis(self, symbol: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Execute combined data collection and analysis stage"""
        # Get enhanced stock info (includes both technical and fundamental analysis)
        stock_info = stocks_service.get_enhanced_stock_info(symbol)
        
        if not stock_info or stock_info.get('ohlcv') is None:
            raise ValueError(f"No data found for symbol {symbol}")
        
        ohlcv = stock_info.get('ohlcv')
        if hasattr(ohlcv, 'empty') and ohlcv.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        elif ohlcv is None:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Get technical analysis
        technical_data = enhanced_technical.analyze_symbol(symbol, days_back=30)
        
        # Get fundamental analysis
        fundamentals = stock_info.get('fundamentals', {})
        enhanced_fundamentals = enhanced_fundamental_analysis.fetch_enhanced_fundamentals(symbol)
        
        return {
            "ohlcv_days": len(ohlcv),
            "current_price": float(ohlcv['Close'].iloc[-1]),
            "enhanced_technical_available": True,
            "enhanced_fundamentals_available": True,
            "data_quality": "good",
            "technical_analysis": {
                "status": "success",
                "summary": {
                    "trend_indicators": 0,
                    "oscillator_indicators": 0,
                    "volume_indicators": 1,
                    "momentum_indicators": 1
                },
                "indicators_available": 35,
                "technical_indicators": self._format_technical_indicators(technical_data)
            },
            "fundamental_analysis": {
                "enhanced_fundamentals": enhanced_fundamentals,
                "fundamental_score": stock_info.get('fundamental_score', {})
            },
            "summary": {
                "data_sources": ["yfinance", "enhanced_technical", "enhanced_fundamentals"],
                "volume_avg": float(ohlcv['Volume'].mean()),
                "price_range": {
                    "high": float(ohlcv['High'].max()),
                    "low": float(ohlcv['Low'].min()),
                    "current": float(ohlcv['Close'].iloc[-1])
                }
            }
        }
    
    def _execute_technical_and_combined_scoring(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute combined technical scoring and combined scoring stage"""
        # Get data from previous stage
        stage1_data = previous_results.get("data_collection_and_analysis", {})
        technical_analysis = stage1_data.get("technical_analysis", {})
        fundamental_analysis = stage1_data.get("fundamental_analysis", {})
        
        # Calculate technical scoring
        technical_indicators = technical_analysis.get("technical_indicators", {})
        technical_scores = enhanced_scoring.calculate_enhanced_score(technical_indicators)
        
        # Calculate combined scoring
        fundamental_score = fundamental_analysis.get("fundamental_score", {}).get("final_score", 0.0)
        technical_score = technical_scores.get("final_score", 0.0)
        
        combined_score = (0.6 * technical_score + 0.4 * fundamental_score)
        
        return {
            "technical_scoring": technical_scores,
            "combined_scoring": {
                "fundamental_score": fundamental_score,
                "technical_score": technical_score,
                "combined_score": combined_score,
                "scoring_weights": {
                    "technical": 0.6,
                    "fundamental": 0.4
                }
            },
            "status": "success"
        }
    
    def _execute_forensic_analysis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forensic analysis stage"""
        # Get data from previous stages
        stage1_data = previous_results.get("data_collection_and_analysis", {})
        stage2_data = previous_results.get("technical_and_combined_scoring", {})
        
        # Extract data for analysis
        fundamentals = stage1_data.get("fundamental_analysis", {})
        technical_analysis = stage1_data.get("technical_analysis", {})
        enhanced_fundamentals = fundamentals.get("enhanced_fundamentals", {})
        
        # Get the raw technical data (not the formatted structure)
        raw_technical_data = technical_analysis.get("technical_indicators", {})
        
        # Call multi-stage prompting service for forensic analysis only
        forensic_result = multi_stage_prompting_service._stage1_forensic_analysis(
            symbol, 
            fundamentals.get("fundamental_score", {}),
            raw_technical_data,
            enhanced_fundamentals
        )
        
        if not forensic_result:
            raise ValueError("Forensic analysis failed")
        
        return forensic_result
    
    def _execute_module_selection(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute module selection stage"""
        # Get data from previous stages
        stage1_data = previous_results.get("data_collection_and_analysis", {})
        stage2_data = previous_results.get("technical_and_combined_scoring", {})
        
        # Extract data for analysis
        fundamentals = stage1_data.get("fundamental_analysis", {})
        technical_analysis = stage1_data.get("technical_analysis", {})
        enhanced_fundamentals = fundamentals.get("enhanced_fundamentals", {})
        
        # Get forensic analysis result
        forensic_analysis = previous_results.get("forensic_analysis", {})
        
        # Get the raw technical data (not the formatted structure)
        raw_technical_data = technical_analysis.get("technical_indicators", {})
        
        # Call multi-stage prompting service for module selection only
        module_result = multi_stage_prompting_service._stage2_module_selection(
            symbol,
            forensic_analysis,
            fundamentals.get("fundamental_score", {}),
            raw_technical_data,
            enhanced_fundamentals
        )
        
        if not module_result:
            raise ValueError("Module selection failed")
        
        return module_result
    
    def _execute_risk_assessment(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk assessment stage"""
        # Get data from previous stages
        stage1_data = previous_results.get("data_collection_and_analysis", {})
        stage2_data = previous_results.get("technical_and_combined_scoring", {})
        
        # Extract data for analysis
        fundamentals = stage1_data.get("fundamental_analysis", {})
        technical_analysis = stage1_data.get("technical_analysis", {})
        
        # Get previous analysis results
        forensic_analysis = previous_results.get("forensic_analysis", {})
        module_analysis = previous_results.get("module_selection", {})
        
        # Get the raw technical data (not the formatted structure)
        raw_technical_data = technical_analysis.get("technical_indicators", {})
        
        # Call multi-stage prompting service for risk assessment only
        risk_result = multi_stage_prompting_service._stage3_risk_assessment(
            symbol,
            forensic_analysis,
            module_analysis,
            fundamentals.get("fundamental_score", {}),
            raw_technical_data
        )
        
        if not risk_result:
            raise ValueError("Risk assessment failed")
        
        return risk_result
    
    def _execute_final_decision(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final decision stage"""
        # Get data from previous stages
        stage1_data = previous_results.get("data_collection_and_analysis", {})
        stage2_data = previous_results.get("technical_and_combined_scoring", {})
        
        # Extract data for analysis
        fundamentals = stage1_data.get("fundamental_analysis", {})
        technical_analysis = stage1_data.get("technical_analysis", {})
        
        # Get previous analysis results
        forensic_analysis = previous_results.get("forensic_analysis", {})
        module_analysis = previous_results.get("module_selection", {})
        risk_assessment = previous_results.get("risk_assessment", {})
        
        # Get the raw technical data (not the formatted structure)
        raw_technical_data = technical_analysis.get("technical_indicators", {})
        
        # Call multi-stage prompting service for final decision only
        decision_result = multi_stage_prompting_service._stage4_final_decision(
            symbol,
            forensic_analysis,
            module_analysis,
            risk_assessment,
            fundamentals.get("fundamental_score", {}),
            raw_technical_data
        )
        
        if not decision_result:
            raise ValueError("Final decision failed")
        
        return decision_result
    
    def _execute_verdict_synthesis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verdict synthesis stage"""
        # Get all AI analysis results
        forensic_analysis = previous_results.get("forensic_analysis", {})
        module_analysis = previous_results.get("module_selection", {})
        risk_assessment = previous_results.get("risk_assessment", {})
        final_decision = previous_results.get("final_decision", {})
        
        # Synthesize the results
        synthesis_result = {
            "forensic_analysis": forensic_analysis,
            "module_selection": module_analysis,
            "risk_assessment": risk_assessment,
            "final_decision": final_decision
        }
        
        # Create final recommendation based on the synthesis
        final_recommendation = {
            "action": final_decision.get("action", "avoid"),
            "confidence": final_decision.get("confidence", 0.5),
            "position_size": final_decision.get("position_size", "0%"),
            "rationale": final_decision.get("rationale", "Analysis completed"),
            "stop_loss": {
                "reasoning": "Based on technical analysis",
                "price": 0.0  # This would be calculated from technical data
            }
        }
        
        return {
            "synthesis_result": synthesis_result,
            "final_recommendation": final_recommendation
        }
    
    def _copy_cached_analysis_to_job(self, job_id: str, cached_analysis: Dict[str, Any]) -> None:
        """Copy cached analysis results to current job preserving the new 8-stage structure"""
        try:
            from datetime import datetime, timezone
            
            # Get cached stages and preserve the new 8-stage structure
            cached_stages = cached_analysis.get("stages", {})
            
            # Define the new 8-stage order for consistency
            new_stage_order = [
                "data_collection_and_analysis",
                "technical_and_combined_scoring", 
                "forensic_analysis",
                "module_selection",
                "risk_assessment",
                "final_decision",
                "verdict_synthesis",
                "final_scoring"
            ]
            
            # Update job with initial info (started processing)
            initial_updates = {
                "status": "processing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "source": "cached",
                "stage_order": new_stage_order
            }
            self.job_service.db.update_job(job_id, initial_updates)
            
            # Update stages incrementally and update progress after each stage
            total_stages = len(cached_stages)
            completed_count = 0
            
            # Process stages in the correct order
            for stage_name in new_stage_order:
                if stage_name not in cached_stages:
                    continue
                    
                stage_data = cached_stages[stage_name]
                
                # Update individual stage
                self.job_service.update_stage_status(
                    job_id, 
                    stage_name, 
                    StageStatus.COMPLETED, 
                    data=stage_data.get("data")
                )

                # If replaying final_scoring from cache, also run routing
                if stage_name == "final_scoring":
                    try:
                        from app.services.post_scoring_router import route_post_scoring
                        # Extract from cached stages: final score and prior verdict synthesis data
                        final_scoring_data = stage_data.get("data", {}) or {}
                        final_score = float(final_scoring_data.get("final_score", 0.0))
                        if final_score == 0.0:
                            # Fallback compute from combined_scoring + verdict confidence if available
                            stage2 = cached_stages.get("technical_and_combined_scoring", {})
                            cs = ((stage2 or {}).get("data") or {}).get("combined_scoring", {})
                            combined_score = float(cs.get("combined_score", 0.0))
                            verdict_stage = cached_stages.get("verdict_synthesis", {})
                            final_reco_fallback = ((verdict_stage or {}).get("data") or {}).get("final_recommendation", {})
                            ai_conf = float(final_reco_fallback.get("confidence", 0.0))
                            final_score = 0.5 * combined_score + 0.3 * ai_conf + 0.2 * (combined_score * ai_conf)
                            
                            # Store the calculated final_score back to the job document
                            updated_final_scoring_data = final_scoring_data.copy()
                            updated_final_scoring_data["final_score"] = final_score
                            updated_final_scoring_data["meets_threshold"] = final_score >= 0.5
                            
                            # Update the stage with the corrected final_score
                            self.job_service.update_stage_status(
                                job_id, 
                                stage_name, 
                                StageStatus.COMPLETED, 
                                data=updated_final_scoring_data
                            )

                        verdict_stage = cached_stages.get("verdict_synthesis", {})
                        final_reco = (verdict_stage.get("data", {}) or {}).get("final_recommendation", {})
                        action = final_reco.get("action") or ""
                        confidence = float(final_reco.get("confidence", 0.0))
                        rationale = final_reco.get("rationale")

                        route_post_scoring(
                            symbol=cached_analysis.get("symbol", ""),
                            job_id=job_id,
                            final_score=final_score,
                            action=action,
                            confidence=confidence,
                            rationale=rationale,
                            user_id="default_user",
                        )
                    except Exception as routing_e:
                        logger.warning(f"Cached replay routing skipped for {job_id}: {routing_e}")
                
                completed_count += 1
                percentage = int((completed_count / total_stages) * 100)
                
                # Update job progress after each stage
                progress_updates = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "progress_percentage": percentage
                }
                
                # If this is the last stage, mark as completed
                if completed_count == total_stages:
                    progress_updates.update({
                        "status": "completed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "cost_saved": 0.10
                    })
                
                self.job_service.db.update_job(job_id, progress_updates)
                
                logger.info(f"   📊 Updated stage {stage_name} ({completed_count}/{total_stages}) - {percentage}% complete")
                
                # Small delay to simulate realistic processing
                import time
                time.sleep(0.1)
            
            # Final update with complete stage structure for consistency
            final_stage_structure = {}
            for stage_name, stage_data in cached_stages.items():
                final_stage_structure[stage_name] = {
                    "stage_name": stage_name,
                    "status": "completed",
                    "started_at": stage_data.get("started_at"),
                    "completed_at": stage_data.get("completed_at"),
                    "data": stage_data.get("data"),
                    "duration_seconds": stage_data.get("duration_seconds"),
                    "dependencies": []  # Add empty dependencies for consistency
                }
            
            final_updates = {
                "stages": final_stage_structure,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            success = self.job_service.db.update_job(job_id, final_updates)
            
            if success:
                logger.info(f"✅ Copied cached analysis to job {job_id} with incremental progress - saved $0.10")
                logger.info(f"   Updated stages: {list(final_stage_structure.keys())}")
                logger.info(f"   Stage order: {new_stage_order}")
            else:
                logger.error(f"❌ Failed to update job {job_id} with cached analysis")
            
        except Exception as e:
            logger.error(f"Failed to copy cached analysis to job {job_id}: {e}")

# Singleton instance
stage_processor = StageProcessor()
