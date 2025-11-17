"""
FastAPI routes for recommendations endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional

from app.models.schemas import (
    ApiResponse, Recommendation, RecommendationCreateRequest, 
    RecommendationUpdateRequest, RecommendationsListResponse, DecisionType
)
from pydantic import BaseModel
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

class RecommendationActionRequest(BaseModel):
    action: str  # APPROVE, REJECT, ADD_TO_WATCHLIST
    recommendation_id: str
    user_id: Optional[str] = "default_user"

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.post("/recommendations", response_model=ApiResponse)
async def create_recommendation(request: RecommendationCreateRequest):
    """Create a new recommendation"""
    try:
        recommendation_data = request.model_dump()
        recommendation_id = firestore_client.create_recommendation(recommendation_data)
        
        return ApiResponse(
            ok=True,
            data={"id": recommendation_id, "message": "Recommendation created successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to create recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{recommendation_id}", response_model=ApiResponse)
async def get_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Get a specific recommendation by ID"""
    try:
        recommendation = firestore_client.get_recommendation(recommendation_id)
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        return ApiResponse(ok=True, data=recommendation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/recommendations/{recommendation_id}", response_model=ApiResponse)
async def update_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID"),
    request: RecommendationUpdateRequest = None
):
    """Update a recommendation"""
    try:
        # Check if recommendation exists
        existing = firestore_client.get_recommendation(recommendation_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Update only provided fields
        updates = {}
        if request.status is not None:
            updates["status"] = request.status.value
        if request.reason is not None:
            updates["reason"] = request.reason
        if request.priority is not None:
            updates["priority"] = request.priority.value
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        success = firestore_client.update_recommendation(recommendation_id, updates)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update recommendation")
        
        return ApiResponse(ok=True, data={"message": "Recommendation updated successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations", response_model=ApiResponse)
async def list_recommendations(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected"),
    limit: int = Query(50, ge=1, le=100, description="Number of recommendations to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor")
):
    """List recommendations with optional filtering"""
    try:
        # Use default_user if user_id is not provided
        if not user_id:
            user_id = "default_user"
            
        result = firestore_client.list_recommendations(
            user_id=user_id,
            status=status,
            limit=limit,
            cursor=cursor
        )
        
        return ApiResponse(ok=True, data=result)
    except Exception as e:
        logger.error(f"Failed to list recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{recommendation_id}/trade-details", response_model=ApiResponse)
async def get_trade_details(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Get detailed trade information for a recommendation"""
    try:
        # Import here to avoid circular imports
        from app.services.trade_details_calculator import trade_details_calculator
        
        # Get recommendation
        recommendation = firestore_client.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Get source job ID
        source_job_id = recommendation.get('source_job_id')
        if not source_job_id:
            raise HTTPException(status_code=400, detail="No source job ID found for this recommendation")
        
        # Get job analysis data
        job_analysis = firestore_client.get_job_analysis_data(source_job_id)
        if not job_analysis:
            raise HTTPException(status_code=404, detail="Analysis data not found for this recommendation")
        
        # Calculate trade details
        trade_details = trade_details_calculator.calculate_trade_details(recommendation, job_analysis)
        
        # Check for errors in calculation
        if trade_details.get('error'):
            raise HTTPException(status_code=500, detail=f"Error calculating trade details: {trade_details['error']}")
        
        return ApiResponse(ok=True, data=trade_details)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade details for recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/recommendations/{recommendation_id}", response_model=ApiResponse)
async def delete_recommendation(
    recommendation_id: str = Path(..., description="Recommendation ID")
):
    """Delete a recommendation"""
    try:
        # Check if recommendation exists
        existing = firestore_client.get_recommendation(recommendation_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        success = firestore_client.delete_recommendation(recommendation_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete recommendation")
        
        return ApiResponse(ok=True, data={"message": "Recommendation deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete recommendation {recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/recommendations", response_model=ApiResponse)
async def get_user_recommendations(
    user_id: str = Path(..., description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of recommendations to return")
):
    """Get recommendations for a specific user"""
    try:
        result = firestore_client.list_recommendations(
            user_id=user_id,
            status=status,
            limit=limit
        )
        
        return ApiResponse(ok=True, data=result)
    except Exception as e:
        logger.error(f"Failed to get recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/action", response_model=ApiResponse)
async def action_recommendation(
    request: RecommendationActionRequest
):
    """
    Take action on a recommendation:
    - APPROVE: Add to portfolio and mark as approved
    - REJECT: Mark as rejected
    - ADD_TO_WATCHLIST: Add to watchlist and mark as approved
    """
    try:
        action = request.action.upper()
        recommendation_id = request.recommendation_id
        user_id = request.user_id
        
        # Get recommendation
        recommendation = firestore_client.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        symbol = recommendation.get('symbol')
        recommendation_action = recommendation.get('action')  # buy/sell
        
        # Validate action
        if action not in ['APPROVE', 'REJECT', 'ADD_TO_WATCHLIST']:
            raise HTTPException(status_code=400, detail="Invalid action. Must be: APPROVE, REJECT, or ADD_TO_WATCHLIST")
        
        # Create user decision record
        decision_data = {
            "recommendation_id": recommendation_id,
            "decision": action,
            "user_id": user_id,
            "symbol": symbol,
            "action_taken": action.lower()
        }
        decision_id = firestore_client.create_user_decision(decision_data)
        
        # Execute action
        if action == 'APPROVE':
            # Create position with snapshot
            from app.services.position_service import position_service
            position_result = position_service.convert_recommendation_to_position(
                recommendation_id, user_id
            )
            
            # Also add to portfolio for backward compatibility
            await _add_to_portfolio_from_recommendation(recommendation, user_id)
            
            firestore_client.update_recommendation(recommendation_id, {"status": "approved"})
            message = f"Recommendation approved. Position {position_result['position_id']} created with snapshot"
            
        elif action == 'REJECT':
            firestore_client.update_recommendation(recommendation_id, {"status": "rejected"})
            message = "Recommendation rejected"
            
        elif action == 'ADD_TO_WATCHLIST':
            # Add to watchlist
            watchlist_data = {
                "user_id": user_id,
                "symbol": symbol,
                "reason": f"From recommendation: {recommendation_action}",
                "source": "recommendation_action",
                "source_id": recommendation_id
            }
            firestore_client.add_to_watchlist(watchlist_data)
            firestore_client.update_recommendation(recommendation_id, {"status": "approved"})
            message = f"Recommendation approved and added to watchlist"
        
        return ApiResponse(
            ok=True,
            data={
                "decision_id": decision_id,
                "action": action,
                "symbol": symbol,
                "message": message
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process recommendation action {request.action} for {request.recommendation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _add_to_portfolio_from_recommendation(recommendation: dict, user_id: str):
    """Add recommendation to portfolio using stored or calculated trade details"""
    try:
        # Try to use stored trade details first
        stored_trade_details = recommendation.get('trade_details', {})
        
        if stored_trade_details and stored_trade_details.get('entry_price'):
            # Use stored trade details
            quantity = stored_trade_details.get('quantity', 0)
            entry_price = stored_trade_details.get('entry_price', 0.0)
            current_price = stored_trade_details.get('current_price', entry_price)  # Fallback to entry price
            action = recommendation.get('action', 'buy').lower()
            
            logger.info(f"Using stored trade details for recommendation {recommendation.get('id')}")
        else:
            # Fall back to calculating trade details
            logger.info(f"Calculating trade details for recommendation {recommendation.get('id')}")
            from app.services.trade_details_calculator import trade_details_calculator
            
            source_job_id = recommendation.get('source_job_id')
            if not source_job_id:
                logger.warning(f"No source job ID for recommendation {recommendation.get('id')}")
                return
            
            job_analysis = firestore_client.get_job_analysis_data(source_job_id)
            if not job_analysis:
                logger.warning(f"No job analysis data for {source_job_id}")
                return
            
            trade_details_result = trade_details_calculator.calculate_trade_details(recommendation, job_analysis)
            if trade_details_result.get('error'):
                logger.error(f"Error calculating trade details: {trade_details_result['error']}")
                return
            
            position_sizing = trade_details_result.get('trade_details', {}).get('position_sizing', {})
            entry_details = trade_details_result.get('trade_details', {}).get('entry', {})
            
            quantity = position_sizing.get('recommended_shares', 0)
            entry_price = entry_details.get('target_entry', 0.0)
            current_price = entry_details.get('current_price', 0.0)
            action = recommendation.get('action', 'buy').lower()
        
        # Calculate P&L based on action type
        if action == 'buy':
            # For BUY: Profit when current_price > entry_price
            pnl = (current_price - entry_price) * quantity
        elif action == 'sell':
            # For SELL: Profit when current_price < entry_price (short position)
            pnl = (entry_price - current_price) * quantity
        else:
            # Default to buy logic
            pnl = (current_price - entry_price) * quantity
        
        portfolio_data = {
            "user_id": user_id,
            "symbol": recommendation.get('symbol'),
            "quantity": quantity,
            "avg_price": entry_price,  # Price per share at entry
            "entry_date": recommendation.get('created_at'),
            "current_price": current_price,  # Current price per share
            "current_value": current_price * quantity,  # Total current value
            "invested_amount": entry_price * quantity,  # Total amount invested
            "pnl": pnl,  # Profit/Loss (corrected for action type)
            "status": "active",
            "source": "recommendation",
            "source_id": recommendation.get('id'),
            "action": recommendation.get('action')
        }
        
        portfolio_id = firestore_client.create_portfolio_item(portfolio_data)
        logger.info(f"Added recommendation {recommendation.get('id')} to portfolio as {portfolio_id}")
        
    except Exception as e:
        logger.error(f"Failed to add recommendation to portfolio: {e}")
        # Don't raise - this is a helper function
