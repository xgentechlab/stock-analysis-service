"""
FastAPI routes for signals endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.schemas import ApiResponse, SignalsListResponse, Signal
from app.db.firestore_client import firestore_client
from app.models.firestore_models import dict_to_signal
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/signals", tags=["signals"])

@router.get("", response_model=ApiResponse)
async def list_signals(
    status: Optional[str] = Query("open", description="Filter by status: open, all"),
    limit: int = Query(50, ge=1, le=100, description="Number of signals to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor")
):
    """
    List signals with optional filtering
    - status: open (default), all, placed, dismissed, expired
    - limit: number of results (1-100, default 50)
    - cursor: pagination cursor for next page
    """
    try:
        result = firestore_client.list_signals(status=status, limit=limit, cursor=cursor)
        
        # Convert to Signal models
        signals = []
        for signal_data in result["signals"]:
            try:
                signal = dict_to_signal(signal_data)
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Error converting signal data: {e}")
                continue
        
        response_data = SignalsListResponse(
            signals=signals,
            total=result["total"],
            cursor=result.get("cursor")
        )
        
        return ApiResponse(ok=True, data=response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error listing signals: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/{signal_id}", response_model=ApiResponse)
async def get_signal(signal_id: str):
    """Get a specific signal by ID"""
    try:
        signal_data = firestore_client.get_signal(signal_id)
        
        if not signal_data:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        signal = dict_to_signal(signal_data)
        
        return ApiResponse(ok=True, data=signal.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal {signal_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.post("/{signal_id}/place", response_model=ApiResponse)
async def place_signal_order(signal_id: str, order_request: dict):
    """
    Place an order for a signal
    Creates position and fill records, updates signal status
    """
    try:
        from app.models.schemas import PlaceOrderRequest
        from app.services.order_service import order_service
        
        # Validate request
        try:
            place_order_req = PlaceOrderRequest(**order_request)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
        
        # Get signal
        signal_data = firestore_client.get_signal(signal_id)
        if not signal_data:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Check signal status
        if signal_data.get("status") != "open":
            raise HTTPException(status_code=400, detail="Signal is not open for trading")
        
        # Check system configuration
        runtime_config = firestore_client.get_runtime_config()
        
        if runtime_config.get("kill_switch", False):
            raise HTTPException(status_code=503, detail="Trading is disabled (kill switch active)")
        
        # Calculate order value and check limits
        current_price = signal_data["technical"]["close"]
        order_value_minor = int(current_price * place_order_req.qty * 100)  # Convert to paise
        
        max_notional = runtime_config.get("max_order_notional_minor", 2000000)
        if order_value_minor > max_notional:
            raise HTTPException(
                status_code=400, 
                detail=f"Order value exceeds maximum limit of ₹{max_notional/100:.2f}"
            )
        
        # Place the order (this will create position and fill)
        result = await order_service.place_order(
            signal_id=signal_id,
            signal_data=signal_data,
            qty=place_order_req.qty,
            stop=place_order_req.stop,
            target=place_order_req.target,
            paper_mode=runtime_config.get("paper_mode", True)
        )
        
        return ApiResponse(ok=True, data=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order for signal {signal_id}: {e}")
        return ApiResponse(ok=False, error=str(e))
