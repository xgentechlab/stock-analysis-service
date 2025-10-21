"""
Post-final-scoring routing service.
Decides whether to create a recommendation (buy/sell) or add to watchlist
based on final score, AI action, and confidence. Single-user system.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _priority_for_buy(score: float, conf: float) -> str:
    if score >= 0.80 and conf >= 0.75:
        return "urgent"
    if score >= 0.70 or conf >= 0.70:
        return "high"
    return "medium"


def _priority_for_sell(score: float, conf: float) -> str:
    if score >= 0.70 or conf >= 0.70:
        return "high"
    return "medium"


def route_post_scoring(
    *,
    symbol: str,
    job_id: str,
    final_score: float,
    action: str,
    confidence: float,
    rationale: Optional[str] = None,
    user_id: str = "default_user",
) -> None:
    """
    Fire-and-forget style routing. Any exception is logged and swallowed to
    avoid impacting the main pipeline.
    """
    try:
        # Lazy import to avoid circular dependencies at import time
        from app.db.firestore_client import firestore_client
        logger.info(
            "[post_scoring_router] Routing decision for symbol=%s job_id=%s action=%s final_score=%.3f confidence=%.3f",
            symbol,
            job_id,
            action,
            final_score,
            confidence,
        )

        # Log rationale and user
        logger.debug(
            "[post_scoring_router] User: %s | Rationale: %s",
            user_id,
            rationale if rationale else "(none provided)",
        )
        norm_action = (action or "").lower()  # buy | watch | avoid
        rationale_text = rationale or "Final decision after enhanced analysis"

        # Recommendations - BUY
        if norm_action == "buy" and final_score >= 0.60 and confidence >= 0.60:
            firestore_client.create_recommendation({
                "user_id": user_id,
                "symbol": symbol,
                "action": "buy",
                "reason": rationale_text,
                "priority": _priority_for_buy(final_score, confidence),
                "source_job_id": job_id,
                "final_score": final_score,
                "confidence": confidence,
            })
            return

        # Recommendations - SELL (avoid)
        if norm_action == "avoid" and final_score >= 0.55 and confidence >= 0.55:
            firestore_client.create_recommendation({
                "user_id": user_id,
                "symbol": symbol,
                "action": "sell",
                "reason": rationale_text,
                "priority": _priority_for_sell(final_score, confidence),
                "source_job_id": job_id,
                "final_score": final_score,
                "confidence": confidence,
            })
            return

        # Watchlist - watch or ambiguous
        ambiguous = (0.45 <= final_score < 0.60) or (confidence < 0.55)
        if norm_action == "watch" or ambiguous:
            firestore_client.add_to_watchlist({
                "user_id": user_id,
                "symbol": symbol,
            })
    except Exception as e:
        logger.warning(f"post_scoring_router: routing failed for {symbol} ({job_id}): {e}")


