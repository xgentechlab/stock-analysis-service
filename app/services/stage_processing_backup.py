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
from app.db.firestore_client import firestore_client

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
            
            # Initialize stage_results
            stage_results = {}
            
            # Get stage execution order (now 4 stages)
            stage_order = self.stage_config.get_stage_order(analysis_type)
            completed_stages = []
            failed_stages = []
            
            # Process stages with parallel execution support
            logger.info(f"🔄 Starting stage processing loop for {symbol}")
            logger.info(f"📋 Total stages to process: {len(stage_order)}")
            logger.info(f"📋 Stage order: {stage_order}")
            logger.info(f"✅ Initial completed stages: {completed_stages}")
            logger.info(f"❌ Initial failed stages: {failed_stages}")
            
            # Use full stage order for progress calculation (4 stages total)
            full_stage_order = self.stage_config.get_stage_order(analysis_type)
            total_stages = len(full_stage_order)
            logger.info(f"📊 Progress calculation: Using full {total_stages} stages for progress")
            
            iteration = 0
            while len(completed_stages) + len(failed_stages) < total_stages:
                iteration += 1
                logger.info(f"🔄 STAGE LOOP ITERATION {iteration} for {symbol}")
                logger.info(f"📊 Current status: {len(completed_stages)} completed, {len(failed_stages)} failed, {total_stages} total")
                
                # Get parallel stages that can be executed (excluding failed stages)
                parallel_stages = self.stage_config.get_parallel_stages(analysis_type, completed_stages)
                logger.info(f"🔄 Parallel stages from config: {parallel_stages}")
                
                # Filter out already failed stages
                parallel_stages = [stage for stage in parallel_stages if stage not in failed_stages]
                logger.info(f"🔄 Parallel stages after filtering: {parallel_stages}")
                
                if not parallel_stages:
                    logger.warning(f"⚠️ No more stages can be executed for {symbol}")
                    logger.warning(f"📊 Final status: {len(completed_stages)} completed, {len(failed_stages)} failed")
                    logger.warning(f"📋 Remaining stages: {[s for s in stage_order if s not in completed_stages and s not in failed_stages]}")
                    break
                
                if len(parallel_stages) == 1:
                    # Single stage - execute normally
                    stage_name = parallel_stages[0]
                    logger.info(f"🎯 EXECUTING SINGLE STAGE: {stage_name} for {symbol}")
                    logger.info(f"📊 Stage execution order: {len(completed_stages) + 1}/{total_stages}")
                    
                    success, result, error = self._execute_stage(
                        job_id, symbol, analysis_type, stage_name, stage_results
                    )
                    
                    if success:
                        completed_stages.append(stage_name)
                        stage_results[stage_name] = result
                        logger.info(f"✅ STAGE SUCCESS: {stage_name} completed successfully")
                        logger.info(f"📊 Updated completed stages: {completed_stages}")
                        logger.info(f"📊 Remaining stages: {[s for s in stage_order if s not in completed_stages and s not in failed_stages]}")
                    else:
                        logger.error(f"❌ STAGE FAILED: {stage_name} failed: {error}")
                        failed_stages.append(stage_name)
                        logger.error(f"📊 Updated failed stages: {failed_stages}")
                        
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
            
            # Log the final stage processing status
            logger.info(f"🏁 STAGE PROCESSING LOOP COMPLETED for {symbol}")
            logger.info(f"📊 Final status: {len(completed_stages)} completed, {len(failed_stages)} failed, {total_stages} total")
            logger.info(f"✅ Completed stages: {completed_stages}")
            logger.info(f"❌ Failed stages: {failed_stages}")
            logger.info(f"📋 Expected stages: {full_stage_order}")
            
            # Check if all expected stages were completed
            missing_stages = [s for s in full_stage_order if s not in completed_stages and s not in failed_stages]
            if missing_stages:
                logger.warning(f"⚠️ MISSING STAGES: {missing_stages}")
                logger.warning(f"⚠️ These stages were not executed: {missing_stages}")
            else:
                logger.info(f"✅ ALL STAGES EXECUTED: All {total_stages} stages were processed")
            
            # Check if job should be marked as failed due to too many failed stages
            if failed_stages and len(failed_stages) >= total_stages * 0.5:  # More than 50% failed
                logger.error(f"❌ TOO MANY FAILED STAGES: ({len(failed_stages)}/{total_stages}). Failing job.")
                self._mark_job_as_failed(job_id, f"Too many failed stages: {failed_stages}")
                return False
            
            logger.info(f"🎉 JOB PROCESSING COMPLETED for {symbol}")
            logger.info(f"📊 Final result: {len(completed_stages)} completed, {len(failed_stages)} failed")
            
            # Mark job as completed in database
            self._mark_job_as_completed(job_id, completed_stages, failed_stages)
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
            "simple_analysis",  # Now includes data collection
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
    
    def _mark_job_as_completed(self, job_id: str, completed_stages: list, failed_stages: list) -> None:
        """Mark a job as completed with final results"""
        try:
            from app.models.schemas import JobStatus
            from datetime import datetime, timezone
            
            # Update job status to completed
            updates = {
                "status": JobStatus.COMPLETED.value,
                "completed_stages": completed_stages,
                "failed_stages": failed_stages,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "progress_percentage": 100,
                "cost_saved": 0.10
            }
            
            self.job_service.db.update_job(job_id, updates)
            logger.info(f"Job {job_id} marked as completed with {len(completed_stages)} completed stages")
            
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as completed: {e}")
    
    def _get_initial_stage_data(self, stage_name: str, symbol: str) -> Dict[str, Any]:
        """Get initial data to show while stage is processing"""
        initial_data = {
            "status": "processing",
            "message": f"Starting {stage_name.replace('_', ' ').title()} for {symbol}",
            "progress": 0,
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add stage-specific initial data
        if stage_name == "simple_analysis":
            initial_data.update({
                "message": f"Performing comprehensive analysis for {symbol} (includes data collection and scoring)",
                "steps": ["Fetching OHLCV data", "Getting fundamentals", "Enhanced technical analysis", "Enhanced fundamental analysis", "Technical scoring", "Combined scoring"]
            })
        elif stage_name == "simple_decision":
            initial_data.update({
                "message": f"Making trading decision for {symbol}",
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
            if stage_name == "simple_analysis":
                result = self._execute_simple_analysis(symbol, previous_results)
            elif stage_name == "simple_decision":
                result = self._execute_simple_decision(symbol, previous_results)
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

                    logger.info(f"🔍 STAGE_EXECUTION: Routing data for {symbol}")
                    logger.info(f"  - synthesis keys: {list(synthesis.keys()) if synthesis else 'None'}")
                    logger.info(f"  - final_reco keys: {list(final_reco.keys()) if final_reco else 'None'}")
                    logger.info(f"  - action: {action}")
                    logger.info(f"  - confidence: {confidence}")
                    logger.info(f"  - rationale: {rationale}")
                    logger.info(f"  - final_score: {final_score}")

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
    

    
    def _execute_final_scoring(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final scoring stage"""
        logger.info(f"🔍 FINAL_SCORING: Starting for {symbol}")
        
        # Correct sources: combined score from simple_analysis, confidence from verdict synthesis
        simple_analysis = previous_results.get("simple_analysis", {}) or {}
        # Get combined score from simple_analysis (now includes technical and fundamental scoring)
        combined_score = float(simple_analysis.get("combined_score", 0.0))

        verdict_stage = previous_results.get("verdict_synthesis", {}) or {}
        final_reco = verdict_stage.get("final_recommendation", {}) or {}
        ai_confidence = float(final_reco.get("confidence", 0.0))
        
        logger.info(f"🔍 FINAL_SCORING: Input data for {symbol}")
        logger.info(f"  - combined_score: {combined_score}")
        logger.info(f"  - verdict_stage keys: {list(verdict_stage.keys()) if verdict_stage else 'None'}")
        logger.info(f"  - final_reco keys: {list(final_reco.keys()) if final_reco else 'None'}")
        logger.info(f"  - ai_confidence: {ai_confidence}")
        
        # Fallback: try final_decision if verdict_synthesis missing
        if ai_confidence == 0.0:
            logger.warning(f"⚠️ FINAL_SCORING: ai_confidence is 0.0, trying fallback for {symbol}")
            fd = previous_results.get("final_decision", {}) or {}
            if isinstance(fd, dict):
                ai_confidence = float(fd.get("confidence", ai_confidence))
                logger.info(f"  - fallback ai_confidence: {ai_confidence}")
        
        # SIMPLIFIED FINAL SCORE: 60% combined + 40% confidence
        # This makes the scoring more transparent and predictable
        final_score = 0.6 * combined_score + 0.4 * ai_confidence
        
        logger.info(f"🔍 FINAL_SCORING: Calculated final_score for {symbol}")
        logger.info(f"  - final_score: {final_score}")
        logger.info(f"  - meets_threshold: {final_score >= 0.5}")
        logger.info(f"  - calculation: 0.6 * {combined_score} + 0.4 * {ai_confidence} = {final_score}")
        
        return {
            "final_score": final_score,
            "meets_threshold": final_score >= 0.5,
            "threshold": 0.5,
            "blending_method": "simple_60_40",
            "blending_weights": {
                "combined_score": 0.6,
                "llm_confidence": 0.4
            },
            "components": {
                "combined_contribution": 0.6 * combined_score,
                "confidence_contribution": 0.4 * ai_confidence
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
        critical_stages = ["simple_analysis", "verdict_synthesis"]
        return stage_name in critical_stages
    
    # New stage execution methods for the 8-stage structure
    
    
    def _execute_simple_analysis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simple analysis stage (replaces forensic_analysis)"""
        # FIX: Get database data from job analysis instead of previous_results
        db_data = self._get_database_data_from_job(symbol, previous_results)
        
        # Extract required inputs from database data
        fundamental_score = db_data.get("fundamental_score", {})
        raw_technical_data = db_data.get("raw_technical_data", {})
        enhanced_fundamentals = db_data.get("enhanced_fundamentals", {})
        
        # Call new 2-stage method for simple analysis
        simple_analysis_result = multi_stage_prompting_service._stage1_simple_analysis(
            symbol, 
            fundamental_score,
            raw_technical_data,
            enhanced_fundamentals
        )
        
        if not simple_analysis_result:
            raise ValueError("Simple analysis failed")
        
        # CRITICAL FIX: Calculate combined_score from technical_score and fundamental_score
        # This is needed for final_scoring stage
        fundamental_score = float(simple_analysis_result.get("fundamental_score", 0.0))
        technical_score = float(simple_analysis_result.get("technical_score", 0.0))
        
        # Balanced combined_score: 50% technical + 50% fundamental
        # This gives equal weight to both technical and fundamental analysis
        combined_score = 0.5 * technical_score + 0.5 * fundamental_score
        simple_analysis_result["combined_score"] = combined_score
        
        logger.info(f"✅ Calculated combined_score for {symbol}: {combined_score} (technical: {technical_score} * 0.5, fundamental: {fundamental_score} * 0.5)")
        
        return simple_analysis_result
    
    def _execute_simple_decision(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simple decision stage (replaces module_selection, risk_assessment, final_decision)"""
        # FIX: Get database data from job analysis instead of previous_results
        db_data = self._get_database_data_from_job(symbol, previous_results)
        
        # Get simple analysis result from previous stage
        simple_analysis = previous_results.get("simple_analysis", {})
        
        # Extract required inputs from database data
        fundamental_score = db_data.get("fundamental_score", {})
        raw_technical_data = db_data.get("raw_technical_data", {})
        enhanced_fundamentals = db_data.get("enhanced_fundamentals", {})
        
        # Call new 2-stage method for simple decision
        simple_decision_result = multi_stage_prompting_service._stage2_simple_decision(
            symbol,
            simple_analysis,
            fundamental_score,
            raw_technical_data,
            enhanced_fundamentals
        )
        
        if not simple_decision_result:
            raise ValueError("Simple decision failed")
        
        return simple_decision_result
    
    
    def _execute_verdict_synthesis(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute verdict synthesis stage"""
        logger.info(f"🔍 VERDICT_SYNTHESIS: Starting for {symbol}")
        
        # Get all AI analysis results (compatibility layer ensures these exist)
        forensic_analysis = previous_results.get("forensic_analysis", {})
        module_analysis = previous_results.get("module_selection", {})
        risk_assessment = previous_results.get("risk_assessment", {})
        final_decision = previous_results.get("final_decision", {})
        
        # Also get the new 2-stage results for reference
        simple_analysis = previous_results.get("simple_analysis", {})
        simple_decision = previous_results.get("simple_decision", {})
        
        logger.info(f"🔍 VERDICT_SYNTHESIS: Input data for {symbol}")
        logger.info(f"  - simple_analysis keys: {list(simple_analysis.keys()) if simple_analysis else 'None'}")
        logger.info(f"  - simple_decision keys: {list(simple_decision.keys()) if simple_decision else 'None'}")
        logger.info(f"  - final_decision keys: {list(final_decision.keys()) if final_decision else 'None'}")
        
        # Synthesize the results
        synthesis_result = {
            "forensic_analysis": forensic_analysis,
            "module_selection": module_analysis,
            "risk_assessment": risk_assessment,
            "final_decision": final_decision
        }
        
        # Create final recommendation based on the synthesis
        # FIX: Use simple_decision data instead of final_decision for new 2-stage system
        # The data is directly in simple_decision, not in simple_decision.data
        simple_decision_data = simple_decision.get("data", {}) if simple_decision.get("data") else simple_decision
        
        logger.info(f"🔍 VERDICT_SYNTHESIS: simple_decision_data for {symbol}")
        logger.info(f"  - simple_decision_data keys: {list(simple_decision_data.keys()) if simple_decision_data else 'None'}")
        
        if simple_decision_data:
            decision = simple_decision_data.get("decision", "avoid")
            confidence = simple_decision_data.get("confidence", 0.5)
            position_size = simple_decision_data.get("position_size", "0%")
            reasoning = simple_decision_data.get("reasoning", "Analysis completed")
            stop_loss = simple_decision_data.get("stop_loss", 0.0)
            
            logger.info(f"🔍 VERDICT_SYNTHESIS: Extracted values for {symbol}")
            logger.info(f"  - decision: {decision}")
            logger.info(f"  - confidence: {confidence}")
            logger.info(f"  - position_size: {position_size}")
            logger.info(f"  - reasoning: {reasoning}")
            logger.info(f"  - stop_loss: {stop_loss}")
        else:
            logger.warning(f"⚠️ VERDICT_SYNTHESIS: No simple_decision_data found for {symbol}")
            decision = "avoid"
            confidence = 0.5
            position_size = "0%"
            reasoning = "Analysis completed"
            stop_loss = 0.0
        
        final_recommendation = {
            "action": decision.lower(),
            "confidence": confidence,
            "position_size": position_size,
            "rationale": reasoning,
            "stop_loss": {
                "reasoning": "Based on technical analysis",
                "price": stop_loss
            }
        }
        
        logger.info(f"🔍 VERDICT_SYNTHESIS: Final recommendation for {symbol}")
        logger.info(f"  - action: {final_recommendation['action']}")
        logger.info(f"  - confidence: {final_recommendation['confidence']}")
        logger.info(f"  - position_size: {final_recommendation['position_size']}")
        logger.info(f"  - rationale: {final_recommendation['rationale']}")
        
        return {
            "synthesis_result": synthesis_result,
            "final_recommendation": final_recommendation
        }
    
    def _fetch_comprehensive_db_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch all required database data for complete analysis pipeline"""
        try:
            # Use list methods to avoid index requirements
            hot_analyses = firestore_client.list_hot_stock_analyses(symbol=symbol, limit=1)
            mtf_analyses = firestore_client.list_multi_timeframe_analyses(symbol=symbol, limit=1)
            
            hot_analysis = hot_analyses[0] if hot_analyses else None
            mtf_analysis = mtf_analyses[0] if mtf_analyses else None
            
            if not hot_analysis or not mtf_analysis:
                logger.warning(f"No database data found for {symbol}")
                return {}
            
            # Stage 1 Data: Data Collection and Analysis
            stage1_data = {
                "ohlcv_days": 30,  # Default value since we removed timeframes data
                "current_price": hot_analysis.get("current_price"),
                "enhanced_technical_available": True,
                "enhanced_fundamentals_available": True,
                "data_quality": "good",
                "technical_analysis": {
                    "status": "success",
                    "summary": {
                        "trend_indicators": 4,
                        "oscillator_indicators": 2,
                        "volume_indicators": 5,
                        "momentum_indicators": 9
                    },
                    "indicators_available": 37,
                    "technical_indicators": hot_analysis.get("enhanced_technical_indicators", {})
                },
                "fundamental_analysis": {
                    "enhanced_fundamentals": hot_analysis.get("enhanced_fundamentals", {}),
                    "fundamental_score": hot_analysis.get("enhanced_fundamental_score", {})
                },
                "summary": {
                    "data_sources": ["database", "enhanced_technical", "enhanced_fundamentals"],
                    "volume_avg": mtf_analysis.get("volume_analysis", {}).get("1d", {}).get("avg_volume", 0),
                    "price_range": {
                        "high": hot_analysis.get("current_price", 0) * 1.1,  # Approximate high
                        "low": hot_analysis.get("current_price", 0) * 0.9,   # Approximate low
                        "current": hot_analysis.get("current_price")
                    }
                }
            }
            
            # Stage 2 Data: Technical and Combined Scoring
            # Get scores from the scores field in hot stock analysis
            scores = hot_analysis.get("scores", {})
            technical_score = scores.get("enhanced_technical_score", 0.0)
            technical_confidence = scores.get("enhanced_technical_confidence", 0.0)
            technical_strength = scores.get("enhanced_technical_strength", "unknown")
            fundamental_score = scores.get("enhanced_fundamental_score", 0.0)
            combined_score = scores.get("enhanced_combined_score", 0.0)
            
            # Use MTF scores as fallback if hot stock scores are missing
            if not technical_score:
                technical_score = mtf_analysis.get("mtf_scores", {}).get("overall_score", 0.0)
            if not technical_confidence:
                technical_confidence = mtf_analysis.get("mtf_scores", {}).get("confidence", 0.0)
            if not technical_strength or technical_strength == "unknown":
                technical_strength = mtf_analysis.get("mtf_scores", {}).get("strength", "unknown")
            if not fundamental_score:
                fundamental_score = 0.5  # Default fallback
            if not combined_score:
                combined_score = 0.6 * technical_score + 0.4 * fundamental_score
            
            stage2_data = {
                "technical_scoring": {
                    "final_score": technical_score,
                    "confidence": technical_confidence,
                    "strength": technical_strength
                },
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
            
            return {
                # Raw database data (keep for reference)
                "hot_analysis": hot_analysis,
                "mtf_analysis": mtf_analysis,
                
                # Note: Stage 1 & 2 data now integrated into simple_analysis
                
                # Direct access to components (ensure they are dictionaries, not floats)
                "fundamental_score": scores.get("enhanced_fundamental_score", {}) if isinstance(scores.get("enhanced_fundamental_score"), dict) else {"score": scores.get("enhanced_fundamental_score", 0.0)},
                "combined_score": combined_score,
                "technical_confidence": technical_confidence,
                "technical_strength": technical_strength,
                
                # Technical indicators (single source of truth - use enhanced version from MTF analysis)
                "raw_technical_data": mtf_analysis.get("technical_indicators", {}) or hot_analysis.get("technical_indicators", {}).get("enhanced_technical_indicators", {}),
                
                # MTF data (single source of truth - reference from mtf_analysis)
                "mtf_scores": mtf_analysis.get("mtf_scores", {}),
                "divergence_signals": mtf_analysis.get("divergence_signals", {})
                
                # REMOVED DUPLICATIONS:
                # - enhanced_fundamentals: Available in stage1_data.fundamental_analysis.enhanced_fundamentals
                # - trend_analysis: Available in mtf_analysis.trend_analysis
                # - momentum_scores: Available in mtf_analysis.momentum_scores  
                # - volume_analysis: Available in mtf_analysis.volume_analysis
            }
        except Exception as e:
            logger.error(f"Failed to fetch comprehensive database data for {symbol}: {e}")
            return {}
    
    def _is_db_data_sufficient(self, db_data: Dict[str, Any]) -> bool:
        """Check if database data is sufficient for complete analysis pipeline"""
        # Check for essential data components (now integrated into simple_analysis)
        if not db_data.get("hot_analysis") or not db_data.get("mtf_analysis"):
            return False
        
        # Check for critical components
        required_fields = [
            "fundamental_score", "raw_technical_data", "enhanced_fundamentals",
            "combined_score", "technical_confidence"
        ]
        
        return all(db_data.get(field) for field in required_fields)
    
    def _copy_cached_analysis_to_job(self, job_id: str, cached_analysis: Dict[str, Any]) -> None:
        """Copy cached analysis results to current job preserving the new 8-stage structure"""
        try:
            from datetime import datetime, timezone
            
            # Get cached stages and preserve the new 8-stage structure
            cached_stages = cached_analysis.get("stages", {})
            
            # Define the new 4-stage order for consistency (simplified pipeline)
            new_stage_order = [
                "simple_analysis",
                "simple_decision",
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
            # Use the actual stages being processed, not the cached stages
            total_stages = len(new_stage_order)
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
                            # Fallback compute from simple_analysis + verdict confidence if available
                            simple_analysis = cached_stages.get("simple_analysis", {})
                            combined_score = float(((simple_analysis or {}).get("data") or {}).get("combined_score", 0.0))
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
    
    def _store_database_data_in_job(self, job_id: str, db_data: Dict[str, Any]) -> bool:
        """Store database_data in job analysis for AI stages to access"""
        try:
            from app.db.firestore_client import firestore_client
            
            # Get current job analysis data
            current_analysis = firestore_client.get_job_analysis_data(job_id)
            if not current_analysis:
                logger.error(f"No job analysis data found for {job_id}")
                return False
            
            # Update the stages with database_data
            stages = current_analysis.get("stages", {})
            stages["database_data"] = db_data
            
            # Update the job analysis data in database
            success = firestore_client.update_job_analysis_data(job_id, {"stages": stages})
            
            if success:
                logger.info(f"✅ Stored database_data in job {job_id} analysis")
            else:
                logger.error(f"❌ Failed to store database_data in job {job_id} analysis")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing database_data in job {job_id}: {e}")
            return False
    
    def _get_database_data_from_job(self, symbol: str, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get database_data from job analysis for AI stages"""
        try:
            # First try to get from previous_results (for backward compatibility)
            db_data = previous_results.get("database_data", {})
            if db_data:
                logger.debug(f"Using database_data from previous_results for {symbol}")
                return db_data
            
            # If not in previous_results, get from job analysis
            from app.db.firestore_client import firestore_client
            
            # Get job_id from previous_results or find it
            job_id = previous_results.get("job_id")
            if not job_id:
                # Try to find job by symbol
                jobs = firestore_client.list_jobs(limit=1)
                for job in jobs:
                    if job.get("symbol") == symbol:
                        job_id = job.get("job_id")
                        break
            
            if job_id:
                current_analysis = firestore_client.get_job_analysis_data(job_id)
                if current_analysis:
                    stages = current_analysis.get("stages", {})
                    db_data = stages.get("database_data", {})
                    if db_data:
                        logger.debug(f"Retrieved database_data from job analysis for {symbol}")
                        return db_data
            
            # Fallback to empty dict
            logger.warning(f"No database_data found for {symbol}, using empty data")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting database_data for {symbol}: {e}")
            return {}

# Singleton instance
stage_processor = StageProcessor()
