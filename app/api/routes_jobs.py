"""
Job API routes for async analysis pipeline
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud import firestore

from app.models.schemas import (
    ApiResponse, JobCreateRequest, JobStatusResponse, StageDataResponse, 
    StageMapping, AnalysisType
)
from app.services.job_service import job_service
from app.services.stage_config import stage_config_service
from app.services.stage_processor import stage_processor
from app.config import settings
from app.analysis.utilities.data_formatters import format_optimized_analysis_response

logger = logging.getLogger(__name__)

def _optimize_analysis_response(analysis: dict) -> dict:
    """
    Optimize analysis response by removing duplications and null values
    Reduces response size by 40-60% while maintaining all essential data
    """
    try:
        if not analysis or not isinstance(analysis, dict):
            return analysis
            
        optimized = analysis.copy()
        
        # Remove empty synthesis_result objects that contain only empty dicts
        stages = optimized.get("stages", {})
        for stage_name, stage_data in stages.items():
            if isinstance(stage_data, dict) and "data" in stage_data:
                stage_data_copy = stage_data["data"].copy() if isinstance(stage_data["data"], dict) else {}
                
                # Remove empty synthesis_result
                if "synthesis_result" in stage_data_copy:
                    synthesis_result = stage_data_copy["synthesis_result"]
                    if isinstance(synthesis_result, dict) and all(
                        isinstance(v, dict) and len(v) == 0 
                        for v in synthesis_result.values()
                    ):
                        del stage_data_copy["synthesis_result"]
                        stage_data["data"] = stage_data_copy
                
                # Remove empty objects in final_recommendation
                if "final_recommendation" in stage_data_copy:
                    final_rec = stage_data_copy["final_recommendation"]
                    if isinstance(final_rec, dict):
                        # Remove empty synthesis_result from final_recommendation
                        if "synthesis_result" in final_rec:
                            synthesis_result = final_rec["synthesis_result"]
                            if isinstance(synthesis_result, dict) and all(
                                isinstance(v, dict) and len(v) == 0 
                                for v in synthesis_result.values()
                            ):
                                del final_rec["synthesis_result"]
        
        # Remove zero values in price_range that indicate missing data
        database_data = optimized.get("stages", {}).get("database_data", {})
        if isinstance(database_data, dict):
            data_collection = database_data.get("data_collection_and_analysis", {})
            if isinstance(data_collection, dict):
                summary = data_collection.get("summary", {})
                if isinstance(summary, dict):
                    price_range = summary.get("price_range", {})
                    if isinstance(price_range, dict):
                        # Remove zero values that indicate missing data
                        if price_range.get("high") == 0:
                            del price_range["high"]
                        if price_range.get("low") == 0:
                            del price_range["low"]
        
        logger.debug(f"🔧 Response optimization completed for analysis")
        return optimized
        
    except Exception as e:
        logger.warning(f"Error optimizing analysis response: {e}")
        return analysis

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
        
        # Build complete analysis structure using the new 8-stage structure
        complete_analysis = {
            "symbol": job_data["symbol"],
            "stages": {
                "data_collection_and_analysis": stages.get("data_collection_and_analysis", {}).get("data", {}),
                "technical_and_combined_scoring": stages.get("technical_and_combined_scoring", {}).get("data", {}),
                "forensic_analysis": stages.get("forensic_analysis", {}).get("data", {}),
                "module_selection": stages.get("module_selection", {}).get("data", {}),
                "risk_assessment": stages.get("risk_assessment", {}).get("data", {}),
                "final_decision": stages.get("final_decision", {}).get("data", {}),
                "verdict_synthesis": stages.get("verdict_synthesis", {}).get("data", {}),
                "final_scoring": stages.get("final_scoring", {}).get("data", {})
            },
            "summary": {
                "data_collection_ok": bool(stages.get("data_collection_and_analysis", {}).get("data")),
                "technical_scoring_ok": bool(stages.get("technical_and_combined_scoring", {}).get("data")),
                "forensic_analysis_ok": bool(stages.get("forensic_analysis", {}).get("data")),
                "module_selection_ok": bool(stages.get("module_selection", {}).get("data")),
                "risk_assessment_ok": bool(stages.get("risk_assessment", {}).get("data")),
                "final_decision_ok": bool(stages.get("final_decision", {}).get("data")),
                "verdict_synthesis_ok": bool(stages.get("verdict_synthesis", {}).get("data")),
                "final_scoring_ok": bool(stages.get("final_scoring", {}).get("data")),
                "overall_success": job_data.get("status") == "completed",
                "data_points": stages.get("data_collection_and_analysis", {}).get("data", {}).get("ohlcv_days", 0),
                "enhanced_analysis": True,
                "scoring_method": "enhanced_8_stage_analysis_v1"
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
        
        logger.info(f"🔍 ANALYSIS REQUEST: Getting analysis for {symbol} (type: {analysis_type}, check_freshness: {check_freshness})")
        
        # Check Firestore connection first
        if not firestore_client.db:
            logger.error(f"❌ FIRESTORE ERROR: Firestore client not initialized for {symbol}")
            return ApiResponse(
                ok=False, 
                error="Database connection not available",
                data={"symbol": symbol, "analysis_type": analysis_type}
            )
        
        logger.info(f"✅ FIRESTORE: Connected to project {settings.firestore_project_id}, database {settings.firestore_database_id}")
        
        # First, let's check what jobs exist for this symbol
        logger.info(f"🔍 DEBUGGING: Checking all jobs for symbol {symbol}")
        try:
            all_jobs_query = (firestore_client.db.collection("jobs")
                            .where(filter=FieldFilter("symbol", "==", symbol))
                            .order_by("created_at", direction=firestore.Query.DESCENDING)
                            .limit(10))
            
            all_jobs = list(all_jobs_query.stream())
            logger.info(f"📊 DEBUGGING: Found {len(all_jobs)} total jobs for {symbol}")
            
            for i, job_doc in enumerate(all_jobs):
                job_data = job_doc.to_dict()
                job_id = job_data.get('job_id', 'unknown')
                status = job_data.get('status', 'unknown')
                created_at = job_data.get('created_at', 'unknown')
                job_analysis_type = job_data.get('analysis_type', 'unknown')
                stages_count = len(job_data.get('stages', {}))
                
                logger.info(f"  Job {i+1}: ID={job_id}, Status={status}, Type={job_analysis_type}, Created={created_at}, Stages={stages_count}")
                
                # Check if this job has completed stages
                stages = job_data.get('stages', {})
                completed_stages = [name for name, stage in stages.items() if stage.get('status') == 'completed']
                logger.info(f"    Completed stages: {completed_stages}")
                
        except Exception as debug_e:
            logger.error(f"❌ DEBUGGING ERROR: Failed to query jobs for {symbol}: {debug_e}")
        
        # Now try to get the latest analysis
        logger.info(f"🔍 ANALYSIS: Attempting to get latest analysis for {symbol} (type: {analysis_type})")
        analysis = firestore_client.get_latest_analysis_by_symbol(symbol, analysis_type)
        
        if not analysis:
            logger.warning(f"⚠️  ANALYSIS NOT FOUND: No completed analysis found for {symbol} (type: {analysis_type})")
            
            # Let's also check if there are any jobs with different statuses
            try:
                pending_jobs_query = (firestore_client.db.collection("jobs")
                                    .where(filter=FieldFilter("symbol", "==", symbol))
                                    .where(filter=FieldFilter("analysis_type", "==", analysis_type))
                                    .where(filter=FieldFilter("status", "==", "processing")))
                
                pending_jobs = list(pending_jobs_query.stream())
                if pending_jobs:
                    logger.info(f"📋 FOUND PENDING: {len(pending_jobs)} processing jobs for {symbol}")
                
                failed_jobs_query = (firestore_client.db.collection("jobs")
                                   .where(filter=FieldFilter("symbol", "==", symbol))
                                   .where(filter=FieldFilter("analysis_type", "==", analysis_type))
                                   .where(filter=FieldFilter("status", "==", "failed")))
                
                failed_jobs = list(failed_jobs_query.stream())
                if failed_jobs:
                    logger.info(f"❌ FOUND FAILED: {len(failed_jobs)} failed jobs for {symbol}")
                    
            except Exception as status_e:
                logger.error(f"❌ STATUS CHECK ERROR: Failed to check job statuses: {status_e}")
            
            return ApiResponse(
                ok=False, 
                error=f"No completed analysis found for {symbol}",
                data={
                    "symbol": symbol, 
                    "analysis_type": analysis_type,
                    "debug_info": {
                        "total_jobs_found": len(all_jobs) if 'all_jobs' in locals() else 0,
                        "firestore_connected": firestore_client.db is not None,
                        "project_id": settings.firestore_project_id,
                        "database_id": settings.firestore_database_id
                    }
                }
            )
        
        logger.info(f"✅ ANALYSIS FOUND: Retrieved analysis for {symbol}")
        logger.info(f"📊 ANALYSIS DATA: Analysis has {len(analysis.get('stages', {}))} stages")
        
        # Apply enhanced optimization with better data grouping and hidden internal calculations
        optimized_analysis = format_optimized_analysis_response(analysis)
        logger.info(f"🔧 ENHANCED RESPONSE OPTIMIZATION: Applied data grouping and hidden internal calculations")
        
        response_data = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "analysis": optimized_analysis,
            "source": "cached"
        }
        
        # Check freshness if requested
        if check_freshness:
            logger.info(f"🕒 FRESHNESS: Checking freshness for {symbol}")
            freshness_check = firestore_client.is_analysis_fresh_for_investment(analysis)
            response_data["freshness"] = freshness_check
            
            logger.info(f"📅 FRESHNESS RESULT: {freshness_check}")
            
            # Add cost savings info
            if freshness_check["is_fresh"]:
                response_data["cost_saved"] = 0.10  # $0.10 saved by using cache
                response_data["recommendation"] = "use_cached"
                logger.info(f"💰 COST SAVINGS: $0.10 saved using cached analysis for {symbol}")
            else:
                response_data["cost_saved"] = 0.00
                response_data["recommendation"] = "refresh_needed"
                logger.info(f"🔄 REFRESH NEEDED: Analysis for {symbol} is stale")
        
        logger.info(f"✅ SUCCESS: Returning analysis data for {symbol}")
        return ApiResponse(ok=True, data=response_data)
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Failed to get analysis for {symbol}: {e}")
        logger.error(f"❌ ERROR DETAILS: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"❌ STACK TRACE: {traceback.format_exc()}")
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
