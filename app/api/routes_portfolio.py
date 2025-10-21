"""
FastAPI routes for portfolio endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional
from datetime import datetime, timezone

from app.models.schemas import (
    ApiResponse, PortfolioItem, PortfolioCreateRequest, 
    PortfolioUpdateRequest, PortfolioListResponse
)
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["portfolio"])

@router.post("/portfolio", response_model=ApiResponse)
async def create_portfolio_item(request: PortfolioCreateRequest):
    """Create a new portfolio item"""
    try:
        portfolio_data = request.model_dump()
        
        # Set entry_date if not provided
        if not portfolio_data.get("entry_date"):
            portfolio_data["entry_date"] = datetime.now(timezone.utc).isoformat()
        
        portfolio_id = firestore_client.create_portfolio_item(portfolio_data)
        
        return ApiResponse(
            ok=True,
            data={"id": portfolio_id, "message": "Portfolio item created successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to create portfolio item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/{portfolio_id}", response_model=ApiResponse)
async def get_portfolio_item(
    portfolio_id: str = Path(..., description="Portfolio item ID")
):
    """Get a specific portfolio item by ID"""
    try:
        portfolio_item = firestore_client.get_portfolio_item(portfolio_id)
        
        if not portfolio_item:
            raise HTTPException(status_code=404, detail="Portfolio item not found")
        
        return ApiResponse(ok=True, data=portfolio_item)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get portfolio item {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/portfolio/{portfolio_id}", response_model=ApiResponse)
async def update_portfolio_item(
    portfolio_id: str = Path(..., description="Portfolio item ID"),
    request: PortfolioUpdateRequest = None
):
    """Update a portfolio item"""
    try:
        # Check if portfolio item exists
        existing = firestore_client.get_portfolio_item(portfolio_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio item not found")
        
        # Update only provided fields
        updates = {}
        if request.quantity is not None:
            updates["quantity"] = request.quantity
        if request.current_value is not None:
            updates["current_value"] = request.current_value
        if request.pnl is not None:
            updates["pnl"] = request.pnl
        if request.status is not None:
            updates["status"] = request.status.value
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        success = firestore_client.update_portfolio_item(portfolio_id, updates)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update portfolio item")
        
        return ApiResponse(ok=True, data={"message": "Portfolio item updated successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update portfolio item {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio", response_model=ApiResponse)
async def list_portfolio_items(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status: active, closed")
):
    """List portfolio items with optional filtering"""
    try:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        portfolio_items = firestore_client.get_user_portfolio(user_id, status)
        
        return ApiResponse(
            ok=True,
            data={
                "portfolio": portfolio_items,
                "total": len(portfolio_items)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list portfolio items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/portfolio", response_model=ApiResponse)
async def get_user_portfolio(
    user_id: str = Path(..., description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status: active, closed")
):
    """Get user's portfolio"""
    try:
        portfolio_items = firestore_client.get_user_portfolio(user_id, status)
        
        # Calculate portfolio summary
        total_value = 0
        total_pnl = 0
        active_items = 0
        
        for item in portfolio_items:
            if item.get("current_value"):
                total_value += item["current_value"]
            if item.get("pnl"):
                total_pnl += item["pnl"]
            if item.get("status") == "active":
                active_items += 1
        
        return ApiResponse(
            ok=True,
            data={
                "portfolio": portfolio_items,
                "total": len(portfolio_items),
                "summary": {
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "active_items": active_items,
                    "closed_items": len(portfolio_items) - active_items
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to get portfolio for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/portfolio/{portfolio_id}", response_model=ApiResponse)
async def delete_portfolio_item(
    portfolio_id: str = Path(..., description="Portfolio item ID")
):
    """Delete a portfolio item"""
    try:
        # Check if portfolio item exists
        existing = firestore_client.get_portfolio_item(portfolio_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio item not found")
        
        success = firestore_client.delete_portfolio_item(portfolio_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete portfolio item")
        
        return ApiResponse(ok=True, data={"message": "Portfolio item deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete portfolio item {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/{portfolio_id}/close", response_model=ApiResponse)
async def close_portfolio_item(
    portfolio_id: str = Path(..., description="Portfolio item ID"),
    current_value: Optional[float] = Query(None, description="Current value when closing")
):
    """Close a portfolio item (mark as closed)"""
    try:
        # Check if portfolio item exists
        existing = firestore_client.get_portfolio_item(portfolio_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Portfolio item not found")
        
        if existing.get("status") == "closed":
            raise HTTPException(status_code=400, detail="Portfolio item is already closed")
        
        updates = {"status": "closed"}
        
        # Calculate PnL if current_value is provided
        if current_value is not None:
            updates["current_value"] = current_value
            avg_price = existing.get("avg_price", 0)
            quantity = existing.get("quantity", 0)
            if avg_price > 0 and quantity > 0:
                pnl = (current_value - (avg_price * quantity))
                updates["pnl"] = pnl
        
        success = firestore_client.update_portfolio_item(portfolio_id, updates)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to close portfolio item")
        
        return ApiResponse(ok=True, data={"message": "Portfolio item closed successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close portfolio item {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
