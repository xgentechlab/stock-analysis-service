"""
FastAPI routes for portfolio suggestions endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional

from app.models.schemas import (
    ApiResponse, PortfolioSuggestion, PortfolioSuggestionCreateRequest, 
    PortfolioSuggestionUpdateRequest, PortfolioSuggestionsListResponse
)
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["portfolio-suggestions"])

@router.post("/portfolio-suggestions", response_model=ApiResponse)
async def create_portfolio_suggestion(request: PortfolioSuggestionCreateRequest):
    """Create a new portfolio suggestion"""
    try:
        suggestion_data = request.model_dump()
        suggestion_id = firestore_client.create_portfolio_suggestion(suggestion_data)
        
        return ApiResponse(
            ok=True,
            data={"id": suggestion_id, "message": "Portfolio suggestion created successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to create portfolio suggestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio-suggestions/{suggestion_id}", response_model=ApiResponse)
async def get_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID")
):
    """Get a specific portfolio suggestion by ID"""
    try:
        suggestion = firestore_client.get_portfolio_suggestion(suggestion_id)
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        return ApiResponse(ok=True, data=suggestion)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/portfolio-suggestions/{suggestion_id}", response_model=ApiResponse)
async def update_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID"),
    request: PortfolioSuggestionUpdateRequest = None
):
    """Update a portfolio suggestion status"""
    try:
        # Check if suggestion exists
        existing = firestore_client.get_portfolio_suggestion(suggestion_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        # Update status
        updates = {"status": request.status.value}
        
        success = firestore_client.update_portfolio_suggestion(suggestion_id, updates)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update portfolio suggestion")
        
        return ApiResponse(ok=True, data={"message": "Portfolio suggestion updated successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio-suggestions", response_model=ApiResponse)
async def list_portfolio_suggestions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status: pending, accepted, rejected, implemented")
):
    """List portfolio suggestions with optional filtering"""
    try:
        # Use default_user if user_id is not provided
        if not user_id:
            user_id = "default_user"
        
        suggestions = firestore_client.get_user_portfolio_suggestions(user_id, status)
        
        return ApiResponse(
            ok=True,
            data={
                "suggestions": suggestions,
                "total": len(suggestions)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list portfolio suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/portfolio-suggestions", response_model=ApiResponse)
async def get_user_portfolio_suggestions(
    user_id: str = Path(..., description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get portfolio suggestions for a specific user"""
    try:
        suggestions = firestore_client.get_user_portfolio_suggestions(user_id, status)
        
        # Calculate summary statistics
        status_counts = {}
        for suggestion in suggestions:
            suggestion_status = suggestion.get("status", "unknown")
            status_counts[suggestion_status] = status_counts.get(suggestion_status, 0) + 1
        
        return ApiResponse(
            ok=True,
            data={
                "suggestions": suggestions,
                "total": len(suggestions),
                "status_summary": status_counts
            }
        )
    except Exception as e:
        logger.error(f"Failed to get portfolio suggestions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/portfolio-suggestions/{suggestion_id}", response_model=ApiResponse)
async def delete_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID")
):
    """Delete a portfolio suggestion"""
    try:
        # Check if suggestion exists
        existing = firestore_client.get_portfolio_suggestion(suggestion_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        success = firestore_client.delete_portfolio_suggestion(suggestion_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete portfolio suggestion")
        
        return ApiResponse(ok=True, data={"message": "Portfolio suggestion deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio-suggestions/{suggestion_id}/accept", response_model=ApiResponse)
async def accept_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID")
):
    """Accept a portfolio suggestion (convenience endpoint)"""
    try:
        # Check if suggestion exists
        existing = firestore_client.get_portfolio_suggestion(suggestion_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        # Update status to accepted
        success = firestore_client.update_portfolio_suggestion(suggestion_id, {"status": "accepted"})
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to accept portfolio suggestion")
        
        return ApiResponse(
            ok=True,
            data={"message": "Portfolio suggestion accepted successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to accept portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio-suggestions/{suggestion_id}/reject", response_model=ApiResponse)
async def reject_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID")
):
    """Reject a portfolio suggestion (convenience endpoint)"""
    try:
        # Check if suggestion exists
        existing = firestore_client.get_portfolio_suggestion(suggestion_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        # Update status to rejected
        success = firestore_client.update_portfolio_suggestion(suggestion_id, {"status": "rejected"})
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reject portfolio suggestion")
        
        return ApiResponse(
            ok=True,
            data={"message": "Portfolio suggestion rejected successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio-suggestions/{suggestion_id}/implement", response_model=ApiResponse)
async def implement_portfolio_suggestion(
    suggestion_id: str = Path(..., description="Portfolio suggestion ID")
):
    """Mark a portfolio suggestion as implemented (convenience endpoint)"""
    try:
        # Check if suggestion exists
        existing = firestore_client.get_portfolio_suggestion(suggestion_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio suggestion not found")
        
        # Update status to implemented
        success = firestore_client.update_portfolio_suggestion(suggestion_id, {"status": "implemented"})
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to mark portfolio suggestion as implemented")
        
        return ApiResponse(
            ok=True,
            data={"message": "Portfolio suggestion marked as implemented successfully"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to implement portfolio suggestion {suggestion_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
