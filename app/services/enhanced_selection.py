"""
Enhanced Selection Engine
Uses advanced technical analysis and multi-timeframe scoring
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import logging

from app.config import settings
from app.services.stocks import stocks_service
from app.services.enhanced_scoring import enhanced_scoring
from app.services.openai_client import openai_client
from app.services.multi_stage_prompting import multi_stage_prompting_service
from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)

class EnhancedSelectionEngine:
    """Enhanced selection engine with advanced technical analysis"""
    
    def __init__(self):
        self.universe_size = settings.universe_size
        self.daily_candidates = settings.daily_candidates
        self.openai_shortlist = settings.openai_shortlist
        self.min_signal_score = settings.min_signal_score
        self.top_pick_score = settings.top_pick_score
        
        # Enhanced scoring parameters
        self.min_confidence = 0.6  # Minimum confidence for signal creation
        self.require_divergence = False  # Whether to require divergence signals
        self.require_mtf_alignment = True  # Whether to require multi-timeframe alignment
        
        # Fundamental filters
        self.min_market_cap_cr = settings.min_market_cap_cr
        self.max_pe_ratio = settings.max_pe_ratio
        
        # Enhanced scoring weights (including fundamental scoring)
        self.scoring_weights = {
            "technical": 0.60,      # Technical analysis (existing 5 components)
            "fundamental": 0.40     # Fundamental analysis (new 4 components)
        }
    
    def run_enhanced_selection(self) -> Dict[str, Any]:
        """
        Run the enhanced daily stock selection pipeline
        Returns summary of the run
        """
        start_time = time.time()
        
        logger.info("Starting enhanced daily stock selection pipeline")
        
        try:
            # Step 1: Get universe
            universe = stocks_service.get_universe_symbols(self.universe_size)
            logger.info(f"Universe size: {len(universe)} stocks")
            
            # Step 2: Score all candidates with enhanced analysis
            candidates = self._score_enhanced_candidates(universe)
            logger.info(f"Enhanced scored {len(candidates)} candidates")
            
            # Step 3: Filter and rank with enhanced criteria
            filtered_candidates = self._filter_and_rank_enhanced_candidates(candidates)
            logger.info(f"Enhanced filtered to {len(filtered_candidates)} candidates")
            
            # Step 4: Create shortlist for OpenAI
            shortlist = filtered_candidates[:self.openai_shortlist]
            logger.info(f"Shortlist for OpenAI analysis: {len(shortlist)} stocks")
            
            # Step 5: Run OpenAI analysis and create signals
            signals_created = self._create_enhanced_signals_from_shortlist(shortlist)
            
            # Step 6: Prepare summary
            duration = time.time() - start_time
            top_symbols = [candidate['symbol'] for candidate in filtered_candidates[:10]]
            
            summary = {
                "count_in": len(universe),
                "top_symbols": top_symbols,
                "signals_created": signals_created,
                "run_duration_seconds": round(duration, 2),
                "enhanced_analysis": True,
                "scoring_method": "enhanced_multi_timeframe"
            }
            
            logger.info(f"Enhanced selection completed in {duration:.2f}s. Created {signals_created} signals")
            
            # Create audit log
            firestore_client.create_audit_log(
                action="enhanced_daily_selection_completed",
                details=summary,
                source="enhanced_selection_engine"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in enhanced daily selection pipeline: {e}")
            
            # Create error audit log
            firestore_client.create_audit_log(
                action="enhanced_daily_selection_failed",
                details={"error": str(e)},
                source="enhanced_selection_engine"
            )
            
            raise
    
    def _score_enhanced_candidates(self, universe: List[str]) -> List[Dict[str, Any]]:
        """
        Score all candidates using enhanced technical analysis
        Returns list of candidate dictionaries with enhanced scores
        """
        candidates = []
        
        for symbol in universe:
            try:
                # Get enhanced stock info with multi-timeframe analysis
                stock_info = stocks_service.get_enhanced_stock_info(symbol)
                if not stock_info or stock_info.get('ohlcv') is None:
                    logger.debug(f"Skipping {symbol} - no OHLCV data")
                    continue
                
                ohlcv = stock_info['ohlcv']
                fundamentals = stock_info['fundamentals']
                enhanced_technical = stock_info.get('enhanced_technical', {})
                enhanced_fundamentals = stock_info.get('enhanced_fundamentals', {})
                fundamental_score_data = stock_info.get('fundamental_score', {})
                
                # Check fundamental sanity
                if not stocks_service.check_fundamental_sanity(
                    fundamentals, 
                    self.min_market_cap_cr, 
                    self.max_pe_ratio
                ):
                    logger.debug(f"Skipping {symbol} - failed fundamental checks")
                    continue
                
                # Calculate enhanced technical score
                enhanced_score_data = enhanced_scoring.calculate_enhanced_score(enhanced_technical)
                
                if not enhanced_score_data or enhanced_score_data.get("error"):
                    logger.debug(f"Skipping {symbol} - enhanced scoring failed")
                    continue
                
                # Calculate combined score (technical + fundamental)
                technical_score = enhanced_score_data["final_score"]
                fundamental_score = fundamental_score_data.get("final_score", 0.5) if fundamental_score_data else 0.5
                
                # Combined score: 60% technical + 40% fundamental
                combined_score = (
                    self.scoring_weights["technical"] * technical_score +
                    self.scoring_weights["fundamental"] * fundamental_score
                )
                
                # Apply enhanced filtering criteria
                if not self._passes_enhanced_filters(enhanced_score_data, enhanced_technical):
                    logger.debug(f"Skipping {symbol} - failed enhanced filters")
                    continue
                
                candidate = {
                    "symbol": symbol,
                    "enhanced_score": technical_score,  # Keep original technical score
                    "fundamental_score": fundamental_score,
                    "combined_score": combined_score,  # New combined score
                    "signal_strength": enhanced_score_data["signal_strength"],
                    "fundamental_strength": fundamental_score_data.get("fundamental_strength", "average"),
                    "confidence": enhanced_score_data["confidence"],
                    "fundamental_confidence": fundamental_score_data.get("confidence", 0.5),
                    "component_scores": enhanced_score_data["component_scores"],
                    "fundamental_component_scores": fundamental_score_data.get("component_scores", {}),
                    "fundamentals": fundamentals,
                    "enhanced_fundamentals": enhanced_fundamentals,
                    "technical": enhanced_technical,
                    "ohlcv_days": len(ohlcv),
                    "recommendations": enhanced_scoring._get_recommendations(enhanced_score_data, enhanced_technical)
                }
                
                candidates.append(candidate)
                logger.debug(f"{symbol}: combined={combined_score:.3f}, "
                           f"technical={technical_score:.3f}, fundamental={fundamental_score:.3f}, "
                           f"strength={enhanced_score_data['signal_strength']}, "
                           f"confidence={enhanced_score_data['confidence']:.3f}")
                
            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                continue
        
        return candidates
    
    def _passes_enhanced_filters(self, score_data: Dict[str, Any], technical_data: Dict[str, Any]) -> bool:
        """Check if candidate passes enhanced filtering criteria"""
        try:
            # Minimum confidence filter
            confidence = score_data.get("confidence", 0.0)
            if confidence < self.min_confidence:
                return False
            
            # Signal strength filter (only consider buy signals)
            signal_strength = score_data.get("signal_strength", "neutral")
            if signal_strength in ["strong_sell", "sell"]:
                return False
            
            # Divergence requirement (if enabled)
            if self.require_divergence:
                rsi_divergence = technical_data.get("rsi_divergence", {})
                if not (rsi_divergence.get("bullish_divergence") or 
                       rsi_divergence.get("bearish_divergence")):
                    return False
            
            # Multi-timeframe alignment requirement (if enabled)
            if self.require_mtf_alignment:
                mtf_score = score_data.get("component_scores", {}).get("multi_timeframe", 0.5)
                if mtf_score < 0.6:  # Require good multi-timeframe alignment
                    return False
            
            # Volume confirmation requirement
            volume_score = score_data.get("component_scores", {}).get("volume", 0.5)
            if volume_score < 0.4:  # Require some volume confirmation
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced filtering: {e}")
            return False
    
    def _filter_and_rank_enhanced_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and rank candidates by combined score (technical + fundamental) with additional criteria
        """
        # Sort by combined score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        
        # Apply additional ranking criteria
        ranked_candidates = []
        
        for candidate in sorted_candidates:
            # Calculate composite ranking score
            combined_score = candidate['combined_score']
            technical_score = candidate['enhanced_score']
            fundamental_score = candidate['fundamental_score']
            confidence = candidate['confidence']
            fundamental_confidence = candidate.get('fundamental_confidence', 0.5)
            signal_strength = candidate['signal_strength']
            fundamental_strength = candidate.get('fundamental_strength', 'average')
            
            # Boost score for high confidence and strong signals
            ranking_boost = 0.0
            if confidence > 0.8:
                ranking_boost += 0.05  # Technical confidence boost
            if fundamental_confidence > 0.8:
                ranking_boost += 0.05  # Fundamental confidence boost
            if signal_strength in ["strong_buy", "buy"]:
                ranking_boost += 0.05  # Technical signal boost
            if fundamental_strength in ["excellent", "good"]:
                ranking_boost += 0.05  # Fundamental strength boost
            if candidate.get('technical', {}).get('rsi_divergence', {}).get('bullish_divergence'):
                ranking_boost += 0.1  # Divergence bonus
            
            candidate['composite_score'] = combined_score + ranking_boost
            ranked_candidates.append(candidate)
        
        # Re-sort by composite score
        ranked_candidates = sorted(ranked_candidates, key=lambda x: x['composite_score'], reverse=True)
        
        # Log top candidates
        logger.info("Top 10 enhanced candidates by composite score:")
        for i, candidate in enumerate(ranked_candidates[:10]):
            logger.info(f"{i+1}. {candidate['symbol']}: {candidate['composite_score']:.3f} "
                       f"(combined: {candidate['combined_score']:.3f}, "
                       f"technical: {candidate['enhanced_score']:.3f}, "
                       f"fundamental: {candidate['fundamental_score']:.3f}, "
                       f"strength: {candidate['signal_strength']}/{candidate.get('fundamental_strength', 'N/A')})")
        
        return ranked_candidates
    
    def _create_enhanced_signals_from_shortlist(self, shortlist: List[Dict[str, Any]]) -> int:
        """
        Create enhanced signals from shortlist after OpenAI analysis
        Returns number of signals created
        """
        signals_created = 0
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        for candidate in shortlist:
            try:
                symbol = candidate['symbol']
                
                # Check if signal already exists today (idempotency)
                if firestore_client.check_signal_exists_today(symbol, current_date):
                    logger.info(f"Signal for {symbol} already exists today, skipping")
                    continue
                
                # Get multi-stage analysis verdict with enhanced technical and fundamental data
                multi_stage_analysis = multi_stage_prompting_service.analyze_stock(
                    symbol, 
                    candidate['fundamentals'], 
                    candidate['technical'],
                    candidate.get('enhanced_fundamentals', {})
                )
                
                if not multi_stage_analysis or multi_stage_analysis.get("error"):
                    logger.warning(f"Multi-stage analysis failed for {symbol}, skipping signal creation")
                    continue
                
                # Extract verdict from multi-stage analysis
                verdict = multi_stage_analysis.get("final_recommendation", {})
                if not verdict:
                    logger.warning(f"No final recommendation for {symbol}, skipping signal creation")
                    continue
                
                # Calculate final blended score with enhanced weighting
                combined_score = candidate['combined_score']  # Technical + Fundamental
                llm_confidence = verdict['confidence']
                technical_confidence = candidate['confidence']
                fundamental_confidence = candidate.get('fundamental_confidence', 0.5)
                
                # Enhanced blending: 50% combined score, 30% LLM, 20% confidence blend
                confidence_blend = (technical_confidence + fundamental_confidence) / 2
                final_score = (0.5 * combined_score + 
                              0.3 * llm_confidence + 
                              0.2 * confidence_blend)
                
                # Only create signal if meets minimum threshold
                if final_score < self.min_signal_score:
                    logger.info(f"Final enhanced score {final_score:.3f} below threshold for {symbol}")
                    continue
                
                # Create enhanced signal document with multi-stage analysis
                signal_data = self._create_enhanced_signal_document(
                    symbol, candidate, verdict, final_score, multi_stage_analysis
                )
                
                # Save to Firestore
                signal_id = firestore_client.create_signal(signal_data)
                signals_created += 1
                
                # Determine if top pick
                is_top_pick = final_score >= self.top_pick_score
                
                logger.info(f"Created enhanced signal {signal_id} for {symbol} "
                           f"(score: {final_score:.3f}, combined: {combined_score:.3f}, "
                           f"technical: {candidate['enhanced_score']:.3f}, "
                           f"fundamental: {candidate['fundamental_score']:.3f}, "
                           f"confidence: {confidence_blend:.3f}"
                           f"{'*TOP PICK*' if is_top_pick else ''})")
                
            except Exception as e:
                logger.error(f"Error creating enhanced signal for {candidate.get('symbol', 'unknown')}: {e}")
                continue
        
        return signals_created
    
    def _create_enhanced_signal_document(self, symbol: str, candidate: Dict[str, Any], 
                                       verdict: Dict[str, Any], final_score: float, 
                                       multi_stage_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create enhanced signal document for Firestore
        """
        now = datetime.now(timezone.utc)
        expiry = datetime(now.year, now.month, now.day + 1, 9, 0, 0, tzinfo=timezone.utc)
        
        signal_data = {
            "symbol": symbol,
            "venue": "NSE",
            "created_at": now.isoformat(),
            "status": "open",
            "score": final_score,
            "verdict": verdict,
            "fundamentals": candidate['fundamentals'],
            "enhanced_fundamentals": candidate.get('enhanced_fundamentals', {}),
            "technical": candidate['technical'],
            "multi_stage_analysis": multi_stage_analysis or {},
            "meta": {
                "enhanced_score": candidate['enhanced_score'],
                "fundamental_score": candidate['fundamental_score'],
                "combined_score": candidate['combined_score'],
                "confidence": candidate['confidence'],
                "fundamental_confidence": candidate.get('fundamental_confidence', 0.5),
                "signal_strength": candidate['signal_strength'],
                "fundamental_strength": candidate.get('fundamental_strength', 'average'),
                "component_scores": candidate['component_scores'],
                "fundamental_component_scores": candidate.get('fundamental_component_scores', {}),
                "recommendations": candidate['recommendations'],
                "source": "enhanced_daily_scan_v4_multi_stage",
                "ohlcv_days": candidate['ohlcv_days'],
                "scoring_method": "enhanced_technical_fundamental_multi_stage",
                "scoring_weights": self.scoring_weights,
                "analysis_stages": multi_stage_analysis.get("analysis_stages", {}) if multi_stage_analysis else {}
            },
            "expiry": expiry.isoformat()
        }
        
        return signal_data
    
    def get_enhanced_selection_stats(self) -> Dict[str, Any]:
        """Get current enhanced selection engine statistics"""
        return {
            "universe_size": self.universe_size,
            "daily_candidates": self.daily_candidates,
            "openai_shortlist": self.openai_shortlist,
            "min_signal_score": self.min_signal_score,
            "top_pick_score": self.top_pick_score,
            "min_confidence": self.min_confidence,
            "require_divergence": self.require_divergence,
            "require_mtf_alignment": self.require_mtf_alignment,
            "filters": {
                "min_market_cap_cr": self.min_market_cap_cr,
                "max_pe_ratio": self.max_pe_ratio
            },
            "scoring_weights": enhanced_scoring.weights,
            "scoring_thresholds": enhanced_scoring.thresholds
        }

# Singleton instance
enhanced_selection_engine = EnhancedSelectionEngine()
