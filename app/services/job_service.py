"""
Job processing service for async analysis pipeline
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

from app.models.schemas import (
    Job, JobStage, JobStatus, StageStatus, AnalysisType, 
    JobCreateRequest, JobStatusResponse, StageDataResponse
)
from app.db.firestore_client import firestore_client
from app.services.stage_config import stage_config_service

logger = logging.getLogger(__name__)

class JobService:
    """Service for managing analysis jobs and stage processing"""
    
    def __init__(self):
        self.db = firestore_client
        self.stage_config = stage_config_service
    
    def create_job(self, request: JobCreateRequest) -> str:
        """Create a new analysis job"""
        try:
            # Get stage mappings for the analysis type
            stage_mappings = self.stage_config.get_stage_mappings(request.analysis_type)
            
            # Initialize job stages
            stages = {}
            for stage_name, stage_mapping in stage_mappings.items():
                stages[stage_name] = JobStage(
                    stage_name=stage_name,
                    status=StageStatus.PENDING,
                    dependencies=stage_mapping.dependencies
                ).model_dump()
            
            # Create job data
            job_data = {
                "symbol": request.symbol,
                "analysis_type": request.analysis_type.value,
                "status": JobStatus.PENDING.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stages": stages,
                "priority": request.priority
            }
            
            # Create job in database
            job_id = self.db.create_job(job_data)
            logger.info(f"Created job {job_id} for symbol {request.symbol}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create job for {request.symbol}: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        try:
            return self.db.get_job(job_id)
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """Get job status with progress information"""
        try:
            job_data = self.db.get_job(job_id)
            if not job_data:
                return None
            
            # Calculate progress
            stages = job_data.get("stages", {})
            analysis_type = AnalysisType(job_data["analysis_type"])
            
            # Get stage order - use stored order for cached jobs, otherwise use config
            stage_order = job_data.get("stage_order")
            if not stage_order:
                stage_order = self.stage_config.get_stage_order(analysis_type)
            
            # Categorize stages by status
            completed_stages = [name for name, stage in stages.items() 
                              if stage.get("status") == StageStatus.COMPLETED.value]
            
            # Debug logging for job processing
            logger.info(f"🔍 Debug job {job_id} (source: {job_data.get('source', 'fresh')}):")
            logger.info(f"   Stages: {list(stages.keys())}")
            logger.info(f"   Stage statuses: {[(name, stage.get('status')) for name, stage in stages.items()]}")
            logger.info(f"   Stage order: {stage_order}")
            logger.info(f"   Completed stages before sort: {completed_stages}")
            
            # Sort completed stages according to stage order for both cached and regular jobs
            completed_stages = [stage for stage in stage_order if stage in completed_stages]
            
            logger.info(f"   Completed stages after sort: {completed_stages}")
            
            processing_stage = next((name for name, stage in stages.items() 
                                   if stage.get("status") == StageStatus.PROCESSING.value), None)
            
            # Calculate pending stages (all stages after the processing stage)
            pending_stages = []
            if processing_stage:
                # Find the index of the processing stage
                try:
                    processing_index = stage_order.index(processing_stage)
                    # Get all stages after the processing stage
                    pending_stages = stage_order[processing_index + 1:]
                except ValueError:
                    # If processing stage not found in order, use original logic
                    for stage_name in stage_order:
                        if stage_name not in stages or stages[stage_name].get("status") == StageStatus.PENDING.value:
                            stage_deps = stages.get(stage_name, {}).get("dependencies", [])
                            if all(dep in completed_stages for dep in stage_deps):
                                pending_stages.append(stage_name)
            else:
                # If no processing stage, show all stages that haven't started
                for stage_name in stage_order:
                    if stage_name not in stages or stages[stage_name].get("status") == StageStatus.PENDING.value:
                        stage_deps = stages.get(stage_name, {}).get("dependencies", [])
                        if all(dep in completed_stages for dep in stage_deps):
                            pending_stages.append(stage_name)
            
            total_stages = len(stage_order)
            current_stage = self._get_current_stage(stages)
            
            # Build progress object, excluding null values
            progress = {
                "completed_stages_count": len(completed_stages),
                "completed_stages": completed_stages,
                "total_stages": total_stages,
                "percentage": round((len(completed_stages) / total_stages) * 100, 1) if total_stages > 0 else 0,
                "pending_stages": pending_stages,
                "stage_order": stage_order
            }
            
            # Add optional fields only if they have values
            if current_stage:
                progress["current_stage"] = current_stage
            if processing_stage:
                progress["processing_stage"] = processing_stage
            
            # Calculate actual time if completed
            actual_time = None
            if job_data.get("status") == JobStatus.COMPLETED.value and job_data.get("started_at"):
                started = datetime.fromisoformat(job_data["started_at"].replace('Z', '+00:00'))
                completed = datetime.fromisoformat(job_data["completed_at"].replace('Z', '+00:00'))
                actual_time = (completed - started).total_seconds()
            
            # Build response data, excluding null values
            response_data = {
                "job_id": job_data["job_id"],
                "symbol": job_data["symbol"],
                "analysis_type": AnalysisType(job_data["analysis_type"]),
                "status": JobStatus(job_data["status"]),
                "created_at": datetime.fromisoformat(job_data["created_at"].replace('Z', '+00:00')),
                "updated_at": datetime.fromisoformat(job_data["updated_at"].replace('Z', '+00:00')),
                "progress": progress,
                "priority": job_data.get("priority", "normal")
            }
            
            # Add optional fields only if they have values
            if job_data.get("started_at"):
                response_data["started_at"] = datetime.fromisoformat(job_data["started_at"].replace('Z', '+00:00'))
            if job_data.get("completed_at"):
                response_data["completed_at"] = datetime.fromisoformat(job_data["completed_at"].replace('Z', '+00:00'))
            if actual_time is not None:
                response_data["actual_time"] = actual_time
            if job_data.get("error"):
                response_data["error"] = job_data["error"]
            
            return JobStatusResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            raise
    
    def get_stage_data(self, job_id: str, stage_name: str) -> Optional[StageDataResponse]:
        """Get data for a specific stage"""
        try:
            job_data = self.db.get_job(job_id)
            if not job_data:
                return None
            
            stage_data = job_data.get("stages", {}).get(stage_name)
            if not stage_data:
                return None
            
            return StageDataResponse(
                stage_name=stage_data["stage_name"],
                status=StageStatus(stage_data["status"]),
                started_at=datetime.fromisoformat(stage_data["started_at"].replace('Z', '+00:00')) if stage_data.get("started_at") else None,
                completed_at=datetime.fromisoformat(stage_data["completed_at"].replace('Z', '+00:00')) if stage_data.get("completed_at") else None,
                data=stage_data.get("data"),
                error=stage_data.get("error"),
                duration_seconds=stage_data.get("duration_seconds"),
                dependencies=stage_data.get("dependencies", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to get stage data {job_id}/{stage_name}: {e}")
            raise
    
    def update_stage_status(self, job_id: str, stage_name: str, status: StageStatus, 
                          data: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> bool:
        """Update stage status and data"""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Get current stage data
            job_data = self.db.get_job(job_id)
            if not job_data:
                return False
            
            current_stage = job_data.get("stages", {}).get(stage_name, {})
            
            # Calculate duration if completing
            duration_seconds = None
            if status == StageStatus.COMPLETED and current_stage.get("started_at"):
                started = datetime.fromisoformat(current_stage["started_at"].replace('Z', '+00:00'))
                completed = datetime.now(timezone.utc)
                duration_seconds = (completed - started).total_seconds()
            
            # Update stage data
            stage_update = {
                "stage_name": stage_name,
                "status": status.value,
                "started_at": current_stage.get("started_at") or now,
                "completed_at": now if status in [StageStatus.COMPLETED, StageStatus.FAILED] else None,
                "data": data,
                "error": error,
                "duration_seconds": duration_seconds,
                "dependencies": current_stage.get("dependencies", [])
            }
            
            # Update in database
            success = self.db.update_job_stage(job_id, stage_name, stage_update)
            
            if success:
                # Update job status if needed
                self._update_job_status_if_needed(job_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update stage {job_id}/{stage_name}: {e}")
            return False
    
    def _get_current_stage(self, stages: Dict[str, Any]) -> Optional[str]:
        """Get the currently processing stage"""
        for stage_name, stage_data in stages.items():
            if stage_data.get("status") == StageStatus.PROCESSING.value:
                return stage_name
        return None
    
    def _get_current_step_info(self, stages: Dict[str, Any], current_stage: Optional[str]) -> Optional[Dict[str, Any]]:
        """Get current step information for the processing stage"""
        if not current_stage or current_stage not in stages:
            return None
        
        stage_data = stages[current_stage]
        stage_info = stage_data.get("data", {})
        
        # Extract step information from stage data
        if isinstance(stage_info, dict):
            return {
                "stage_name": current_stage,
                "message": stage_info.get("message", f"Processing {current_stage.replace('_', ' ').title()}"),
                "progress": stage_info.get("progress", 0),
                "steps": stage_info.get("steps", []),
                "started_at": stage_data.get("started_at"),
                "duration_seconds": stage_data.get("duration_seconds")
            }
        
        return {
            "stage_name": current_stage,
            "message": f"Processing {current_stage.replace('_', ' ').title()}",
            "progress": 0,
            "steps": [],
            "started_at": stage_data.get("started_at"),
            "duration_seconds": stage_data.get("duration_seconds")
        }
    
    def _update_job_status_if_needed(self, job_id: str) -> None:
        """Update job status based on stage completion"""
        try:
            job_data = self.db.get_job(job_id)
            if not job_data:
                return
            
            stages = job_data.get("stages", {})
            analysis_type = AnalysisType(job_data["analysis_type"])
            
            # Check if all stages are completed
            all_completed = all(
                stage.get("status") == StageStatus.COMPLETED.value 
                for stage in stages.values()
            )
            
            # Check if job is still processing
            is_processing = any(
                stage.get("status") == StageStatus.PROCESSING.value 
                for stage in stages.values()
            )
            
            # Update job status
            updates = {}
            now = datetime.now(timezone.utc).isoformat()
            
            if all_completed:
                updates["status"] = JobStatus.COMPLETED.value
                updates["completed_at"] = now
                
                # Calculate actual time
                if job_data.get("started_at"):
                    started = datetime.fromisoformat(job_data["started_at"].replace('Z', '+00:00'))
                    completed = datetime.now(timezone.utc)
                    updates["actual_time"] = (completed - started).total_seconds()
                    
            elif job_data.get("status") == JobStatus.PENDING.value and is_processing:
                updates["status"] = JobStatus.PROCESSING.value
                updates["started_at"] = now
            
            if updates:
                self.db.update_job(job_id, updates)
                logger.info(f"Updated job {job_id} status: {updates.get('status')}")
                
        except Exception as e:
            logger.error(f"Failed to update job status {job_id}: {e}")

# Singleton instance
job_service = JobService()
