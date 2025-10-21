"""
FastAPI routes for user decisions endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional

from app.models.schemas import (
    ApiResponse, UserDecisionRecord, UserDecisionCreateRequest, UserDecisionsListResponse, DecisionType
)
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["user-decisions"])

@router.post("/user-decisions", response_model=ApiResponse)
async def create_user_decision(request: UserDecisionCreateRequest):
    """Create a new user decision for a recommendation"""
    try:
        # Verify that the recommendation exists
        recommendation = firestore_client.get_recommendation(request.recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        decision_data = request.model_dump()
        decision_id = firestore_client.create_user_decision(decision_data)
        
        # Update the recommendation status based on the decision
        recommendation_updates = {}
        if request.decision.value == "approved":
            recommendation_updates["status"] = "approved"
        elif request.decision.value == "rejected":
            recommendation_updates["status"] = "rejected"
        
        if recommendation_updates:
            firestore_client.update_recommendation(request.recommendation_id, recommendation_updates)
        
        return ApiResponse(
            ok=True,
            data={"id": decision_id, "message": "User decision recorded successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-decisions/{decision_id}", response_model=ApiResponse)
async def get_user_decision(
    decision_id: str = Path(..., description="Decision ID")
):
    """Get a specific user decision by ID"""
    try:
        decision = firestore_client.get_user_decision(decision_id)
        
        if not decision:
            raise HTTPException(status_code=404, detail="User decision not found")
        
        return ApiResponse(ok=True, data=decision)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user decision {decision_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user-decisions", response_model=ApiResponse)
async def list_user_decisions(
    user_id: Optional[str] = Query("default_user", description="Filter by user ID"),
    recommendation_id: Optional[str] = Query(None, description="Filter by recommendation ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of decisions to return")
):
    """List user decisions with optional filtering"""
    try:
        if recommendation_id:
            decisions = firestore_client.get_decisions_for_recommendation(recommendation_id)
        else:
            decisions = firestore_client.list_user_decisions(user_id=user_id, limit=limit)
        
        return ApiResponse(
            ok=True,
            data={
                "decisions": decisions,
                "total": len(decisions)
            }
        )
    except Exception as e:
        logger.error(f"Failed to list user decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{recommendation_id}/decisions", response_model=ApiResponse)
async def get_recommendation_decisions(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Get all decisions for a specific recommendation"""
    try:
        # Verify that the recommendation exists
        recommendation = firestore_client.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        decisions = firestore_client.get_decisions_for_recommendation(recommendation_id)
        
        return ApiResponse(
            ok=True,
            data={
                "recommendation_id": recommendation_id,
                "decisions": decisions,
                "total": len(decisions)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get decisions for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/{recommendation_id}/approve", response_model=ApiResponse)
async def approve_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Approve a recommendation (convenience endpoint)"""
    try:
        # Verify that the recommendation exists
        recommendation = firestore_client.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Create user decision
        decision_data = {
            "recommendation_id": recommendation_id,
            "decision": DecisionType.APPROVED.value
        }
        decision_id = firestore_client.create_user_decision(decision_data)
        
        # Update recommendation status
        firestore_client.update_recommendation(recommendation_id, {"status": "approved"})
        
        return ApiResponse(
            ok=True,
            data={
                "decision_id": decision_id,
                "message": "Recommendation approved successfully"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/{recommendation_id}/reject", response_model=ApiResponse)
async def reject_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Reject a recommendation (convenience endpoint)"""
    try:
        # Verify that the recommendation exists
        recommendation = firestore_client.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Create user decision
        decision_data = {
            "recommendation_id": recommendation_id,
            "decision": DecisionType.REJECTED.value
        }
        decision_id = firestore_client.create_user_decision(decision_data)
        
        # Update recommendation status
        firestore_client.update_recommendation(recommendation_id, {"status": "rejected"})
        
        return ApiResponse(
            ok=True,
            data={
                "decision_id": decision_id,
                "message": "Recommendation rejected successfully"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
