"""
Stocks API routes for CRUD operations on Nifty 500 stocks
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List, Optional, Dict, Any
import logging

from app.models.schemas import (
    Stock, StockCreateRequest, StockUpdateRequest, StockListResponse, 
    ApiResponse
)
from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stocks", tags=["stocks"])

@router.get("/", response_model=ApiResponse)
async def list_stocks(
    industry: Optional[str] = Query(None, description="Filter by industry"),
    is_active: bool = Query(True, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=200, description="Number of stocks to return"),
    offset: int = Query(0, ge=0, description="Number of stocks to skip"),
    search: Optional[str] = Query(None, description="Search by company name or symbol")
):
    """List stocks with optional filtering and pagination"""
    try:
        if search:
            # Use search functionality
            stocks = firestore_client.search_stocks(search, limit=limit)
            total = len(stocks)
        else:
            # Use regular list with filters
            result = firestore_client.list_stocks(
                industry=industry,
                is_active=is_active,
                limit=limit,
                offset=offset
            )
            stocks = result.get("stocks", [])
            total = result.get("total", 0)
        
        return ApiResponse(
            ok=True,
            data={
                "stocks": stocks,
                "total": total,
                "page": (offset // limit) + 1,
                "limit": limit,
                "has_more": (offset + limit) < total
            }
        )
    except Exception as e:
        logger.error(f"Failed to list stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list stocks: {str(e)}")

@router.get("/{stock_id}", response_model=ApiResponse)
async def get_stock(
    stock_id: str = Path(..., description="Stock ID")
):
    """Get a specific stock by ID"""
    try:
        stock = firestore_client.get_stock(stock_id)
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        return ApiResponse(ok=True, data={"stock": stock})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stock {stock_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stock: {str(e)}")

@router.get("/symbol/{symbol}", response_model=ApiResponse)
async def get_stock_by_symbol(
    symbol: str = Path(..., description="Stock symbol")
):
    """Get a specific stock by symbol"""
    try:
        stock = firestore_client.get_stock_by_symbol(symbol)
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        return ApiResponse(ok=True, data={"stock": stock})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stock by symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stock: {str(e)}")

@router.get("/{symbol}/multi-timeframe-analysis", response_model=ApiResponse)
async def get_multi_timeframe_analysis(
    symbol: str = Path(..., description="Stock symbol")
):
    """Get the latest multi-timeframe analysis for a stock"""
    try:
        analysis = firestore_client.get_latest_multi_timeframe_analysis(symbol)
        if not analysis:
            raise HTTPException(status_code=404, detail="Multi-timeframe analysis not found")
        
        return ApiResponse(ok=True, data={"analysis": analysis})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get multi-timeframe analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get multi-timeframe analysis: {str(e)}")

@router.get("/multi-timeframe-analyses", response_model=ApiResponse)
async def list_multi_timeframe_analyses(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(50, ge=1, le=200, description="Number of analyses to return")
):
    """List multi-timeframe analyses with optional filtering"""
    try:
        analyses = firestore_client.list_multi_timeframe_analyses(symbol=symbol, limit=limit)
        
        return ApiResponse(
            ok=True, 
            data={
                "analyses": analyses,
                "total": len(analyses),
                "symbol_filter": symbol
            }
        )
    except Exception as e:
        logger.error(f"Failed to list multi-timeframe analyses: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list multi-timeframe analyses: {str(e)}")

@router.post("/", response_model=ApiResponse)
async def create_stock(
    stock_data: StockCreateRequest
):
    """Create a new stock"""
    try:
        # Check if stock with same symbol already exists
        existing = firestore_client.get_stock_by_symbol(stock_data.symbol)
        if existing:
            raise HTTPException(status_code=400, detail="Stock with this symbol already exists")
        
        # Create stock
        stock_dict = stock_data.model_dump()
        stock_id = firestore_client.create_stock(stock_dict)
        
        # Get the created stock
        created_stock = firestore_client.get_stock(stock_id)
        
        return ApiResponse(
            ok=True,
            data={"stock": created_stock},
            message="Stock created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create stock: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create stock: {str(e)}")

@router.put("/{stock_id}", response_model=ApiResponse)
async def update_stock(
    stock_id: str = Path(..., description="Stock ID"),
    updates: StockUpdateRequest = None
):
    """Update a stock"""
    try:
        # Check if stock exists
        existing_stock = firestore_client.get_stock(stock_id)
        if not existing_stock:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        # Prepare update data (exclude None values)
        update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        # Check for symbol uniqueness if symbol is being updated
        if "symbol" in update_data:
            existing_by_symbol = firestore_client.get_stock_by_symbol(update_data["symbol"])
            if existing_by_symbol and existing_by_symbol["id"] != stock_id:
                raise HTTPException(status_code=400, detail="Stock with this symbol already exists")
        
        # Update stock
        success = firestore_client.update_stock(stock_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update stock")
        
        # Get updated stock
        updated_stock = firestore_client.get_stock(stock_id)
        
        return ApiResponse(
            ok=True,
            data={"stock": updated_stock},
            message="Stock updated successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update stock {stock_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update stock: {str(e)}")

@router.delete("/{stock_id}", response_model=ApiResponse)
async def delete_stock(
    stock_id: str = Path(..., description="Stock ID")
):
    """Delete a stock (soft delete)"""
    try:
        # Check if stock exists
        existing_stock = firestore_client.get_stock(stock_id)
        if not existing_stock:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        # Soft delete stock
        success = firestore_client.delete_stock(stock_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete stock")
        
        return ApiResponse(
            ok=True,
            message="Stock deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete stock {stock_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete stock: {str(e)}")

@router.get("/industry/{industry}", response_model=ApiResponse)
async def get_stocks_by_industry(
    industry: str = Path(..., description="Industry name")
):
    """Get all stocks in a specific industry"""
    try:
        stocks = firestore_client.get_stocks_by_industry(industry)
        
        return ApiResponse(
            ok=True,
            data={
                "stocks": stocks,
                "total": len(stocks),
                "industry": industry
            }
        )
    except Exception as e:
        logger.error(f"Failed to get stocks by industry {industry}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stocks by industry: {str(e)}")

@router.get("/industries/list", response_model=ApiResponse)
async def list_industries():
    """Get list of all industries"""
    try:
        industries = firestore_client.get_all_industries()
        
        return ApiResponse(
            ok=True,
            data={
                "industries": industries,
                "total": len(industries)
            }
        )
    except Exception as e:
        logger.error(f"Failed to get industries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get industries: {str(e)}")

@router.get("/search/{search_term}", response_model=ApiResponse)
async def search_stocks(
    search_term: str = Path(..., description="Search term"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """Search stocks by company name or symbol"""
    try:
        stocks = firestore_client.search_stocks(search_term, limit=limit)
        
        return ApiResponse(
            ok=True,
            data={
                "stocks": stocks,
                "total": len(stocks),
                "search_term": search_term
            }
        )
    except Exception as e:
        logger.error(f"Failed to search stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search stocks: {str(e)}")

@router.get("/stats/summary", response_model=ApiResponse)
async def get_stocks_summary():
    """Get summary statistics of stocks"""
    try:
        # Get all active stocks
        result = firestore_client.list_stocks(is_active=True, limit=10000)
        stocks = result.get("stocks", [])
        
        # Calculate statistics
        total_stocks = len(stocks)
        industries = {}
        
        for stock in stocks:
            industry = stock.get("industry", "Unknown")
            industries[industry] = industries.get(industry, 0) + 1
        
        # Sort industries by count
        sorted_industries = sorted(industries.items(), key=lambda x: x[1], reverse=True)
        
        return ApiResponse(
            ok=True,
            data={
                "total_stocks": total_stocks,
                "total_industries": len(industries),
                "industries_breakdown": dict(sorted_industries[:10]),  # Top 10 industries
                "all_industries": list(industries.keys())
            }
        )
    except Exception as e:
        logger.error(f"Failed to get stocks summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stocks summary: {str(e)}")
