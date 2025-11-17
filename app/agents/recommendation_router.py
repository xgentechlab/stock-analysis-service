"""
Routes agent recommendations to recommendations/watchlists using post_scoring_router
"""
from typing import Dict, Any, Optional
import logging
import uuid

from app.agents.models import FinalRecommendation, Verdict
from app.services.post_scoring_router import route_post_scoring

logger = logging.getLogger(__name__)


def route_agent_recommendation(
    recommendation: FinalRecommendation,
    user_id: str = "default_user"
) -> None:
    """
    Route agent recommendation to recommendations/watchlists
    Maps agent verdicts to post_scoring_router actions
    """
    try:
        from app.db.firestore_client import firestore_client
        
        # Map agent verdict to post_scoring_router action
        action = _map_verdict_to_action(recommendation.recommendation)
        
        # Calculate final score from weighted votes or average agent scores
        final_score = _calculate_final_score(recommendation)
        
        # Generate a unique analysis ID (since we don't have job_id)
        analysis_id = f"agent_analysis_{recommendation.symbol}_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"Routing agent recommendation: {recommendation.symbol} -> "
            f"{action} (score: {final_score:.2f}, confidence: {recommendation.confidence:.2f})"
        )
        
        # Extract trade details if available
        trade_details = recommendation.trade_details or {}
        entry_details = trade_details.get('entry', {})
        exit_details = trade_details.get('exit', {})
        position_sizing = trade_details.get('position_sizing', {})
        risk_metrics = trade_details.get('risk_metrics', {})
        
        # Format trade details for post_scoring_router
        formatted_trade_details = {
            "entry_price": entry_details.get('target_entry', 0.0),
            "exit_price": exit_details.get('target_exit', None),
            "stop_loss": exit_details.get('stop_loss', None),
            "target_price": exit_details.get('target_exit', None),
            "quantity": position_sizing.get('recommended_shares', None),
            "position_size": position_sizing.get('dollar_amount', None),
            "risk_reward_ratio": risk_metrics.get('risk_reward_ratio', None),
            "confidence": recommendation.confidence
        }
        
        # Route to post_scoring_router with trade details
        route_post_scoring(
            symbol=recommendation.symbol,
            job_id=analysis_id,
            final_score=final_score,
            action=action,
            confidence=recommendation.confidence,
            rationale=recommendation.reasoning,
            user_id=user_id,
            trade_details=formatted_trade_details
        )
        
    except Exception as e:
        logger.error(
            f"Failed to route agent recommendation for {recommendation.symbol}: {e}"
        )


def _map_verdict_to_action(verdict: Verdict) -> str:
    """Map agent verdict to post_scoring_router action"""
    mapping = {
        Verdict.BUY: "buy",
        Verdict.SELL: "avoid",  # SELL maps to "avoid" in post_scoring_router
        Verdict.WATCH: "watch",
        Verdict.NEUTRAL: "watch"  # NEUTRAL also goes to watchlist
    }
    return mapping.get(verdict, "watch")


def _calculate_final_score(recommendation: FinalRecommendation) -> float:
    """Calculate final score from recommendation"""
    # Option 1: Use weighted votes to calculate score
    weighted_votes = recommendation.weighted_votes
    total_votes = sum(weighted_votes.values())
    
    if total_votes > 0:
        # Calculate score based on BUY votes vs total
        buy_weight = weighted_votes.get('BUY', 0.0)
        sell_weight = weighted_votes.get('SELL', 0.0)
        
        # Score: positive for BUY, negative for SELL, normalized to 0-1
        net_score = (buy_weight - sell_weight) / max(total_votes, 1.0)
        # Normalize to 0-1 range (where 0.5 is neutral)
        final_score = 0.5 + (net_score * 0.5)
        return max(0.0, min(1.0, final_score))
    
    # Option 2: Fallback to average agent scores
    if recommendation.agent_breakdown:
        avg_score = sum(
            result.score for result in recommendation.agent_breakdown.values()
        ) / len(recommendation.agent_breakdown)
        return avg_score
    
    # Default neutral score
    return 0.5

