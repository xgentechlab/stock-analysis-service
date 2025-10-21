"""
News Intelligence API routes
"""
import logging
from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import ApiResponse, NewsIntelligenceResponse, NewsworthyStock
from app.services.news_intelligence_service import news_intelligence_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["news"])


@router.get("/news/intelligence", response_model=ApiResponse)
async def get_news_intelligence(
    limit: int = Query(20, ge=5, le=30, description="Number of stocks to return")
):
    try:
        results = news_intelligence_service.analyze_news(max_results=limit)
        payload = NewsIntelligenceResponse(
            stocks=[NewsworthyStock(**r) for r in results],
            total=len(results),
        )
        return ApiResponse(ok=True, data=payload.model_dump())
    except Exception as e:
        logger.error(f"Failed to get news intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


