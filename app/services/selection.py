"""
Stock selection algorithm implementation
This is the heart of the system - produces daily shortlist of top signals
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import logging

from app.config import settings
from app.services.stocks import stocks_service
from app.services.indicators import (
    calculate_technical_snapshot, 
    calculate_momentum_score,
    calculate_volume_spike_score,
    calculate_breakout_volatility_score
)
from app.services.openai_client import openai_client
from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)

class SelectionEngine:
    def __init__(self):
        self.universe_size = settings.universe_size
        self.daily_candidates = settings.daily_candidates
        self.openai_shortlist = settings.openai_shortlist
        self.min_signal_score = settings.min_signal_score
        self.top_pick_score = settings.top_pick_score
        
        # Scoring weights
        self.momentum_weight = settings.momentum_weight
        self.volume_weight = settings.volume_weight
        self.breakout_weight = settings.breakout_weight
        
        # Volume parameters
        self.volume_threshold = settings.volume_threshold
        self.volume_cap = settings.volume_cap
        
        # Fundamental filters
        self.min_market_cap_cr = settings.min_market_cap_cr
        self.max_pe_ratio = settings.max_pe_ratio
    
    def run_daily_selection(self) -> Dict[str, Any]:
        """
        Run the complete daily stock selection pipeline
        Returns summary of the run
        """
        start_time = time.time()
        
        logger.info("Starting daily stock selection pipeline")
        
        try:
            # Step 1: Get universe
            universe = stocks_service.get_universe_symbols(self.universe_size)
            logger.info(f"Universe size: {len(universe)} stocks")
            
            # Step 2: Score all candidates
            candidates = self._score_candidates(universe)
            logger.info(f"Scored {len(candidates)} candidates")
            
            # Step 3: Filter and rank
            filtered_candidates = self._filter_and_rank_candidates(candidates)
            logger.info(f"Filtered to {len(filtered_candidates)} candidates")
            
            # Step 4: Create shortlist for OpenAI
            shortlist = filtered_candidates[:self.openai_shortlist]
            logger.info(f"Shortlist for OpenAI analysis: {len(shortlist)} stocks")
            
            # Step 5: Run OpenAI analysis and create signals
            signals_created = self._create_signals_from_shortlist(shortlist)
            
            # Step 6: Prepare summary
            duration = time.time() - start_time
            top_symbols = [candidate['symbol'] for candidate in filtered_candidates[:10]]
            
            summary = {
                "count_in": len(universe),
                "top_symbols": top_symbols,
                "signals_created": signals_created,
                "run_duration_seconds": round(duration, 2)
            }
            
            logger.info(f"Daily selection completed in {duration:.2f}s. Created {signals_created} signals")
            
            # Create audit log
            firestore_client.create_audit_log(
                action="daily_selection_completed",
                details=summary,
                source="selection_engine"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in daily selection pipeline: {e}")
            
            # Create error audit log
            firestore_client.create_audit_log(
                action="daily_selection_failed",
                details={"error": str(e)},
                source="selection_engine"
            )
            
            raise
    
    def _score_candidates(self, universe: List[str]) -> List[Dict[str, Any]]:
        """
        Score all candidates in the universe
        Returns list of candidate dictionaries with scores
        """
        candidates = []
        
        for symbol in universe:
            try:
                # Get stock data
                stock_info = stocks_service.get_stock_info(symbol)
                if not stock_info or stock_info.get('ohlcv') is None:
                    logger.debug(f"Skipping {symbol} - no OHLCV data")
                    continue
                
                ohlcv = stock_info['ohlcv']
                fundamentals = stock_info['fundamentals']
                
                # Check fundamental sanity
                if not stocks_service.check_fundamental_sanity(
                    fundamentals, 
                    self.min_market_cap_cr, 
                    self.max_pe_ratio
                ):
                    logger.debug(f"Skipping {symbol} - failed fundamental checks")
                    continue
                
                # Calculate technical indicators
                technical = calculate_technical_snapshot(ohlcv)
                if not technical:
                    logger.debug(f"Skipping {symbol} - technical calculation failed")
                    continue
                
                # Calculate component scores
                momentum_score = calculate_momentum_score(technical)
                volume_score = calculate_volume_spike_score(
                    technical, self.volume_threshold, self.volume_cap
                )
                breakout_score = calculate_breakout_volatility_score(technical)
                
                # Calculate weighted aggregate score
                raw_score = (
                    self.momentum_weight * momentum_score +
                    self.volume_weight * volume_score +
                    self.breakout_weight * breakout_score
                )
                
                candidate = {
                    "symbol": symbol,
                    "raw_score": raw_score,
                    "momentum_score": momentum_score,
                    "volume_score": volume_score,
                    "breakout_score": breakout_score,
                    "fundamentals": fundamentals,
                    "technical": technical,
                    "ohlcv_days": len(ohlcv)
                }
                
                candidates.append(candidate)
                logger.debug(f"{symbol}: score={raw_score:.3f} (M:{momentum_score:.3f}, V:{volume_score:.3f}, B:{breakout_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                continue
        
        return candidates
    
    def _filter_and_rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and rank candidates by score
        """
        # Sort by raw score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['raw_score'], reverse=True)
        
        # Log top candidates
        logger.info("Top 10 candidates by raw score:")
        for i, candidate in enumerate(sorted_candidates[:10]):
            logger.info(f"{i+1}. {candidate['symbol']}: {candidate['raw_score']:.3f}")
        
        return sorted_candidates
    
    def _create_signals_from_shortlist(self, shortlist: List[Dict[str, Any]]) -> int:
        """
        Create signals from shortlist after OpenAI analysis
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
                
                # Get OpenAI verdict
                verdict = openai_client.get_stock_verdict(
                    symbol, 
                    candidate['fundamentals'], 
                    candidate['technical']
                )
                
                if not verdict:
                    logger.warning(f"No OpenAI verdict for {symbol}, skipping signal creation")
                    continue
                
                # Calculate final blended score
                raw_score = candidate['raw_score']
                llm_confidence = verdict['confidence']
                final_score = 0.5 * raw_score + 0.5 * llm_confidence
                
                # Only create signal if meets minimum threshold
                if final_score < self.min_signal_score:
                    logger.info(f"Final score {final_score:.3f} below threshold for {symbol}")
                    continue
                
                # Create signal document
                signal_data = self._create_signal_document(
                    symbol, candidate, verdict, final_score
                )
                
                # Save to Firestore
                signal_id = firestore_client.create_signal(signal_data)
                signals_created += 1
                
                # Determine if top pick
                is_top_pick = final_score >= self.top_pick_score
                
                logger.info(f"Created signal {signal_id} for {symbol} (score: {final_score:.3f}{'*TOP PICK*' if is_top_pick else ''})")
                
            except Exception as e:
                logger.error(f"Error creating signal for {candidate.get('symbol', 'unknown')}: {e}")
                continue
        
        return signals_created
    
    def _create_signal_document(self, symbol: str, candidate: Dict[str, Any], 
                              verdict: Dict[str, Any], final_score: float) -> Dict[str, Any]:
        """
        Create signal document for Firestore
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
            "technical": candidate['technical'],
            "meta": {
                "trend_score": candidate['raw_score'],
                "source": "daily_scan_v1",
                "momentum_score": candidate['momentum_score'],
                "volume_score": candidate['volume_score'],
                "breakout_score": candidate['breakout_score'],
                "ohlcv_days": candidate['ohlcv_days']
            },
            "expiry": expiry.isoformat()
        }
        
        return signal_data
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get current selection engine statistics"""
        return {
            "universe_size": self.universe_size,
            "daily_candidates": self.daily_candidates,
            "openai_shortlist": self.openai_shortlist,
            "min_signal_score": self.min_signal_score,
            "top_pick_score": self.top_pick_score,
            "weights": {
                "momentum": self.momentum_weight,
                "volume": self.volume_weight,
                "breakout": self.breakout_weight
            },
            "filters": {
                "min_market_cap_cr": self.min_market_cap_cr,
                "max_pe_ratio": self.max_pe_ratio,
                "volume_threshold": self.volume_threshold,
                "volume_cap": self.volume_cap
            }
        }

# Singleton instance
selection_engine = SelectionEngine()
