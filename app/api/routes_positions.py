"""
FastAPI routes for positions endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.models.schemas import ApiResponse, PositionsListResponse, Position
from app.db.firestore_client import firestore_client
from app.models.firestore_models import dict_to_position
from app.services.tracker import position_tracker
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/positions", tags=["positions"])

@router.get("", response_model=ApiResponse)
async def list_positions(status: str = "open"):
    """
    List positions by status
    - status: open (default), closed, all
    """
    try:
        if status == "all":
            # Get both open and closed positions
            open_positions = firestore_client.list_positions(status="open")
            closed_positions = firestore_client.list_positions(status="closed")
            all_positions = open_positions + closed_positions
        else:
            all_positions = firestore_client.list_positions(status=status)
        
        # Convert to Position models
        positions = []
        for position_data in all_positions:
            try:
                position = dict_to_position(position_data)
                positions.append(position)
            except Exception as e:
                logger.warning(f"Error converting position data: {e}")
                continue
        
        response_data = PositionsListResponse(
            positions=positions,
            total=len(positions)
        )
        
        return ApiResponse(ok=True, data=response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error listing positions: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/{position_id}", response_model=ApiResponse)
async def get_position(position_id: str):
    """Get a specific position by ID"""
    try:
        position_data = firestore_client.get_position(position_id)
        
        if not position_data:
            raise HTTPException(status_code=404, detail="Position not found")
        
        position = dict_to_position(position_data)
        
        return ApiResponse(ok=True, data=position.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position {position_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/{position_id}/close", response_model=ApiResponse)
async def close_position(position_id: str):
    """Manually close a position"""
    try:
        success = position_tracker.force_close_position(position_id, "manual_close")
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to close position")
        
        # Get updated position
        position_data = firestore_client.get_position(position_id)
        if position_data:
            position = dict_to_position(position_data)
            return ApiResponse(ok=True, data=position.model_dump())
        else:
            return ApiResponse(ok=True, data={"message": "Position closed successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/summary/stats", response_model=ApiResponse)
async def get_positions_summary():
    """Get summary statistics for all positions"""
    try:
        summary = position_tracker.get_position_summary()
        return ApiResponse(ok=True, data=summary)
        
    except Exception as e:
        logger.error(f"Error getting positions summary: {e}")
        return ApiResponse(ok=False, error=str(e))
