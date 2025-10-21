"""
FastAPI routes for recommendations endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional

from app.models.schemas import (
    ApiResponse, Recommendation, RecommendationCreateRequest, 
    RecommendationUpdateRequest, RecommendationsListResponse
)
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

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
