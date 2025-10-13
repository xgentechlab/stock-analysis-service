"""
Job API routes for async analysis pipeline
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ApiResponse, JobCreateRequest, JobStatusResponse, StageDataResponse, 
    StageMapping, AnalysisType
)
from app.services.job_service import job_service
from app.services.stage_config import stage_config_service
from app.services.stage_processor import stage_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["jobs"])

@router.post("/analyze", response_model=ApiResponse)
async def initiate_analysis(
    request: JobCreateRequest, 
    force_refresh: bool = Query(False, description="Force refresh even if cached analysis exists")
):
    """Initiate a new analysis job with smart caching for investment optimization"""
    try:
        from app.db.firestore_client import firestore_client
        
        logger.info(f"Creating analysis job for {request.symbol} (type: {request.analysis_type.value})")
        
        # Check for cached analysis first (for immediate response)
        cache_info = {
            "cache_available": False,
            "cache_fresh": False,
            "cache_age_days": None,
            "cache_recommendation": "new_analysis",
            "cost_saved": 0.00
        }
        
        if not force_refresh:
            cached_analysis = firestore_client.get_latest_analysis_by_symbol(
                request.symbol, 
                request.analysis_type.value
            )
            
            if cached_analysis:
                freshness_check = firestore_client.is_analysis_fresh_for_investment(cached_analysis)
                
                cache_info.update({
                    "cache_available": True,
                    "cache_fresh": freshness_check["is_fresh"],
                    "cache_age_days": freshness_check["age_days"],
                    "cache_recommendation": freshness_check["recommendation"],
                    "cost_saved": 0.10 if freshness_check["is_fresh"] else 0.00
                })
                
                logger.info(f"🔍 Cache check: {'Fresh' if freshness_check['is_fresh'] else 'Stale'} "
                          f"(age: {freshness_check['age_days']:.1f} days)")
        
        # Create job
        job_id = job_service.create_job(request)
        
        # Start processing in background (asynchronous) - caching logic is handled in stage_processor
        import threading
        
        def run_background_task():
            try:
                success = stage_processor.process_job(job_id)
                if not success:
                    logger.error(f"Job processing failed for {job_id}")
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}")
        
        thread = threading.Thread(target=run_background_task)
        thread.daemon = True
        thread.start()
        
        # Return immediately with job info and cache status
        job_data = job_service.get_job(job_id)
        
        # Determine response message based on cache status
        if cache_info["cache_available"] and cache_info["cache_fresh"]:
            message = f"Analysis job created - will use cached data (saved ${cache_info['cost_saved']:.2f}, age: {cache_info['cache_age_days']:.1f} days)"
        elif cache_info["cache_available"] and not cache_info["cache_fresh"]:
            message = f"Analysis job created - cached data is stale, creating fresh analysis (age: {cache_info['cache_age_days']:.1f} days)"
        else:
            message = "Analysis job created - no cached data found, creating fresh analysis"
        
        return ApiResponse(
            ok=True,
            data={
                "job_id": job_id,
                "symbol": request.symbol,
                "analysis_type": request.analysis_type.value,
                "status": "processing",
                "estimated_time": job_data.get("estimated_time") if job_data else 60,
                "created_at": job_data.get("created_at") if job_data else None,
                "message": message,
                "cache_info": cache_info
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to initiate analysis for {request.symbol}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/jobs/{job_id}/status", response_model=ApiResponse)
async def get_job_status(job_id: str):
    """Get job status and progress"""
    try:
        job_status = job_service.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return ApiResponse(ok=True, data=job_status.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/jobs/{job_id}/stage/{stage_name}", response_model=ApiResponse)
async def get_stage_data(job_id: str, stage_name: str):
    """Get data for a specific stage"""
    try:
        stage_data = job_service.get_stage_data(job_id, stage_name)
        
        if not stage_data:
            raise HTTPException(status_code=404, detail="Stage not found")
        
        return ApiResponse(ok=True, data=stage_data.model_dump())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stage data {job_id}/{stage_name}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/jobs/{job_id}/complete", response_model=ApiResponse)
async def get_complete_analysis(job_id: str):
    """Get complete analysis result"""
    try:
        job_data = job_service.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_data.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Compile complete analysis from all stages
        stages = job_data.get("stages", {})
        
        # Build complete analysis structure similar to the original response
        complete_analysis = {
            "symbol": job_data["symbol"],
            "stages": {
                "data_collection": stages.get("data_collection", {}).get("data", {}),
                "enhanced_technical_analysis": stages.get("enhanced_technical_analysis", {}).get("data", {}),
                "enhanced_technical_scoring": stages.get("enhanced_technical_scoring", {}).get("data", {}),
                "enhanced_fundamental_analysis": stages.get("enhanced_fundamental_analysis", {}).get("data", {}),
                "combined_scoring": stages.get("combined_scoring", {}).get("data", {}),
                "enhanced_filtering": stages.get("enhanced_filtering", {}).get("data", {}),
                "multi_stage_ai_analysis": stages.get("multi_stage_ai_analysis", {}).get("data", {}),
                "final_scoring": stages.get("final_scoring", {}).get("data", {})
            },
            "summary": {
                "fundamental_ok": stages.get("enhanced_fundamental_analysis", {}).get("data", {}).get("fundamental_score", {}).get("final_score", 0) >= 0.5,
                "enhanced_technical_ok": bool(stages.get("enhanced_technical_analysis", {}).get("data")),
                "enhanced_fundamentals_ok": bool(stages.get("enhanced_fundamental_analysis", {}).get("data", {}).get("enhanced_fundamentals")),
                "enhanced_filters_ok": stages.get("enhanced_filtering", {}).get("data", {}).get("passes_enhanced_filters", False),
                "multi_stage_ai_ok": bool(stages.get("multi_stage_ai_analysis", {}).get("data")),
                "ai_verdict_ok": bool(stages.get("multi_stage_ai_analysis", {}).get("data", {}).get("final_recommendation")),
                "overall_success": job_data.get("status") == "completed",
                "data_points": stages.get("data_collection", {}).get("data", {}).get("ohlcv_days", 0),
                "enhanced_analysis": True,
                "scoring_method": "enhanced_technical_fundamental_multi_stage_v4"
            }
        }
        
        return ApiResponse(ok=True, data=complete_analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get complete analysis {job_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/stages", response_model=ApiResponse)
async def get_stage_mappings(analysis_type: Optional[AnalysisType] = Query(None)):
    """Get stage mappings for analysis types"""
    try:
        if analysis_type:
            # Get mappings for specific analysis type
            stages = stage_config_service.get_stage_mappings(analysis_type)
            stage_list = [
                {
                    "stage_name": stage.stage_name,
                    "display_name": stage.display_name,
                    "description": stage.description,
                    "estimated_duration": stage.estimated_duration,
                    "dependencies": stage.dependencies,
                    "order": stage.order
                }
                for stage in stages.values()
            ]
            stage_list.sort(key=lambda x: x["order"])
            
            return ApiResponse(ok=True, data={
                "analysis_type": analysis_type.value,
                "stages": stage_list,
                "total_estimated_time": stage_config_service.get_total_estimated_time(analysis_type)
            })
        else:
            # Get mappings for all analysis types
            all_mappings = {}
            for analysis_type in AnalysisType:
                stages = stage_config_service.get_stage_mappings(analysis_type)
                stage_list = [
                    {
                        "stage_name": stage.stage_name,
                        "display_name": stage.display_name,
                        "description": stage.description,
                        "estimated_duration": stage.estimated_duration,
                        "dependencies": stage.dependencies,
                        "order": stage.order
                    }
                    for stage in stages.values()
                ]
                stage_list.sort(key=lambda x: x["order"])
                
                all_mappings[analysis_type.value] = {
                    "stages": stage_list,
                    "total_estimated_time": stage_config_service.get_total_estimated_time(analysis_type)
                }
            
            return ApiResponse(ok=True, data=all_mappings)
        
    except Exception as e:
        logger.error(f"Failed to get stage mappings: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/jobs", response_model=ApiResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return")
):
    """List jobs with optional status filter"""
    try:
        from app.db.firestore_client import firestore_client
        
        jobs = firestore_client.list_jobs(status=status, limit=limit)
        
        return ApiResponse(ok=True, data={
            "jobs": jobs,
            "count": len(jobs),
            "filters": {
                "status": status,
                "limit": limit
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.delete("/jobs/{job_id}", response_model=ApiResponse)
async def delete_job(job_id: str):
    """Delete a job (for cleanup)"""
    try:
        from app.db.firestore_client import firestore_client
        
        success = firestore_client.delete_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return ApiResponse(ok=True, data={"message": "Job deleted successfully"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/analysis/{symbol}", response_model=ApiResponse)
async def get_analysis_by_symbol(
    symbol: str,
    analysis_type: str = Query("enhanced", description="Analysis type: enhanced or basic"),
    check_freshness: bool = Query(True, description="Check if analysis is fresh for investment")
):
    """Get latest analysis for a symbol with freshness check"""
    try:
        from app.db.firestore_client import firestore_client
        
        logger.info(f"Getting analysis for {symbol} (type: {analysis_type})")
        
        # Get latest analysis
        analysis = firestore_client.get_latest_analysis_by_symbol(symbol, analysis_type)
        
        if not analysis:
            return ApiResponse(
                ok=False, 
                error=f"No completed analysis found for {symbol}",
                data={"symbol": symbol, "analysis_type": analysis_type}
            )
        
        response_data = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "analysis": analysis,
            "source": "cached"
        }
        
        # Check freshness if requested
        if check_freshness:
            freshness_check = firestore_client.is_analysis_fresh_for_investment(analysis)
            response_data["freshness"] = freshness_check
            
            # Add cost savings info
            if freshness_check["is_fresh"]:
                response_data["cost_saved"] = 0.10  # $0.10 saved by using cache
                response_data["recommendation"] = "use_cached"
            else:
                response_data["cost_saved"] = 0.00
                response_data["recommendation"] = "refresh_needed"
        
        return ApiResponse(ok=True, data=response_data)
        
    except Exception as e:
        logger.error(f"Failed to get analysis for {symbol}: {e}")
        return ApiResponse(ok=False, error=str(e))

@router.get("/analysis/{symbol}/history", response_model=ApiResponse)
async def get_analysis_history(
    symbol: str,
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of analyses to return"),
    analysis_type: str = Query("enhanced", description="Analysis type: enhanced or basic")
):
    """Get analysis history for a symbol"""
    try:
        from app.db.firestore_client import firestore_client
        
        logger.info(f"Getting analysis history for {symbol} (last {days_back} days, type: {analysis_type})")
        
        # Get analyses within timeframe
        analyses = firestore_client.get_analysis_by_symbol_and_timeframe(symbol, days_back, analysis_type)
        
        # Limit results
        analyses = analyses[:limit]
        
        # Add freshness info for each analysis
        for analysis in analyses:
            freshness_check = firestore_client.is_analysis_fresh_for_investment(analysis)
            analysis["freshness"] = freshness_check
        
        return ApiResponse(ok=True, data={
            "symbol": symbol,
            "days_back": days_back,
            "total_analyses": len(analyses),
            "analyses": analyses
        })
        
    except Exception as e:
        logger.error(f"Failed to get analysis history for {symbol}: {e}")
        return ApiResponse(ok=False, error=str(e))
