"""
API routes for agentic analysis system
"""
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict
import logging

from app.models.schemas import ApiResponse
from app.agents.market_agent import MarketAgent
from app.agents.storage import agent_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agents"])

market_agent = MarketAgent()


@router.post("/analyze-batch", response_model=ApiResponse)
async def analyze_batch(request: Dict[str, List[str]]):
    """Analyze multiple symbols using agent system"""
    try:
        symbols = request.get("symbols", [])
        if not symbols:
            raise HTTPException(
                status_code=400, 
                detail="Symbols list cannot be empty"
            )
        
        logger.info(f"Analyzing batch of {len(symbols)} symbols")
        
        results = market_agent.analyze_batch(symbols)
        
        successful = []
        failed = []
        
        for symbol, recommendation in results.items():
            try:
                agent_storage.store_final_recommendation(recommendation)
                
                for agent_name, agent_result in recommendation.agent_breakdown.items():
                    agent_storage.store_agent_result(agent_result)
                
                successful.append(symbol)
            except Exception as e:
                logger.error(f"Failed to store results for {symbol}: {e}")
                failed.append(symbol)
        
        return ApiResponse(
            ok=True,
            data={
                "analyzed": len(symbols),
                "successful": len(successful),
                "failed": len(failed),
                "symbols": successful
            }
        )
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}", response_model=ApiResponse)
async def get_analysis(symbol: str):
    """Get full analysis for a symbol"""
    try:
        analysis = agent_storage.get_analysis(symbol)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis found for {symbol}"
            )
        
        return ApiResponse(ok=True, data=analysis)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{symbol}/{agent_name}", response_model=ApiResponse)
async def get_agent_analysis(symbol: str, agent_name: str):
    """Get specific agent analysis"""
    try:
        result = agent_storage.get_agent_result(symbol, agent_name)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No {agent_name} analysis found for {symbol}"
            )
        
        return ApiResponse(ok=True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

