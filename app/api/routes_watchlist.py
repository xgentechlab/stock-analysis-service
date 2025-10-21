"""
FastAPI routes for watchlist endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional

from app.models.schemas import (
    ApiResponse, WatchlistItem, WatchlistAddRequest, WatchlistListResponse
)
from app.db.firestore_client import firestore_client
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["watchlist"])

@router.post("/watchlist", response_model=ApiResponse)
async def add_to_watchlist(request: WatchlistAddRequest):
    """Add a symbol to user's watchlist"""
    try:
        # Check if already in watchlist
        if firestore_client.is_in_watchlist(request.user_id, request.symbol):
            raise HTTPException(
                status_code=400, 
                detail=f"Symbol {request.symbol} is already in user's watchlist"
            )
        
        watchlist_data = request.model_dump()
        doc_id = firestore_client.add_to_watchlist(watchlist_data)
        
        return ApiResponse(
            ok=True,
            data={"id": doc_id, "message": f"Added {request.symbol} to watchlist"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/watchlist/{user_id}/{symbol}", response_model=ApiResponse)
async def remove_from_watchlist(
    user_id: str = Path(..., description="User ID"),
    symbol: str = Path(..., description="Stock symbol")
):
    """Remove a symbol from user's watchlist"""
    try:
        # Check if symbol is in watchlist
        if not firestore_client.is_in_watchlist(user_id, symbol):
            raise HTTPException(
                status_code=404, 
                detail=f"Symbol {symbol} not found in user's watchlist"
            )
        
        success = firestore_client.remove_from_watchlist(user_id, symbol)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove from watchlist")
        
        return ApiResponse(
            ok=True,
            data={"message": f"Removed {symbol} from watchlist"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove from watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watchlist", response_model=ApiResponse)
async def get_default_watchlist():
    """Get default user's watchlist"""
    try:
        watchlist = firestore_client.get_user_watchlist("default_user")
        
        return ApiResponse(
            ok=True,
            data={
                "watchlist": watchlist,
                "total": len(watchlist),
                "user_id": "default_user"
            }
        )
    except Exception as e:
        logger.error(f"Failed to get default watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watchlist/{user_id}", response_model=ApiResponse)
async def get_user_watchlist(
    user_id: str = Path(..., description="User ID")
):
    """Get user's watchlist"""
    try:
        watchlist = firestore_client.get_user_watchlist(user_id)
        
        return ApiResponse(
            ok=True,
            data={
                "watchlist": watchlist,
                "total": len(watchlist)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get watchlist for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watchlist/check/{symbol}", response_model=ApiResponse)
async def check_default_watchlist_status(
    symbol: str = Path(..., description="Stock symbol")
):
    """Check if a symbol is in default user's watchlist"""
    try:
        is_in_watchlist = firestore_client.is_in_watchlist("default_user", symbol)
        
        return ApiResponse(
            ok=True,
            data={
                "symbol": symbol,
                "user_id": "default_user",
                "is_in_watchlist": is_in_watchlist
            }
        )
    except Exception as e:
        logger.error(f"Failed to check default watchlist status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/watchlist/{user_id}/check/{symbol}", response_model=ApiResponse)
async def check_watchlist_status(
    user_id: str = Path(..., description="User ID"),
    symbol: str = Path(..., description="Stock symbol")
):
    """Check if a symbol is in user's watchlist"""
    try:
        is_in_watchlist = firestore_client.is_in_watchlist(user_id, symbol)
        
        return ApiResponse(
            ok=True,
            data={
                "symbol": symbol,
                "is_in_watchlist": is_in_watchlist
            }
        )
    except Exception as e:
        logger.error(f"Failed to check watchlist status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/watchlist/{user_id}/toggle/{symbol}", response_model=ApiResponse)
async def toggle_watchlist(
    user_id: str = Path(..., description="User ID"),
    symbol: str = Path(..., description="Stock symbol")
):
    """Toggle symbol in/out of watchlist"""
    try:
        is_in_watchlist = firestore_client.is_in_watchlist(user_id, symbol)
        
        if is_in_watchlist:
            # Remove from watchlist
            success = firestore_client.remove_from_watchlist(user_id, symbol)
            action = "removed"
        else:
            # Add to watchlist
            watchlist_data = {
                "user_id": user_id,
                "symbol": symbol
            }
            firestore_client.add_to_watchlist(watchlist_data)
            success = True
            action = "added"
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to toggle watchlist")
        
        return ApiResponse(
            ok=True,
            data={
                "message": f"Symbol {symbol} {action} from watchlist",
                "action": action,
                "is_in_watchlist": not is_in_watchlist
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))
