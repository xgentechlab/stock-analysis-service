from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timezone
import logging

from app.config import settings

logger = logging.getLogger(__name__)

class FirestoreClient:
    def __init__(self):
        self.db = None
        try:
            if settings.firestore_project_id:
                self.db = firestore.Client(
                    project=settings.firestore_project_id,
                    database=settings.firestore_database_id
                )
                logger.info(f"Firestore client initialized for project: {settings.firestore_project_id}, database: {settings.firestore_database_id}")
            else:
                logger.warning("Firestore project ID not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            # Don't raise during import - let the application start
            # The error will be handled when methods are called

    def _check_connection(self):
        """Check if Firestore client is properly initialized"""
        if self.db is None:
            raise Exception("Firestore client not initialized. Check your credentials and configuration.")
    
    def _add_meta(self, data: Dict[str, Any], event_id: Optional[str] = None) -> Dict[str, Any]:
        """Add meta block with event_id and created_at to document"""
        meta = {
            "event_id": event_id or str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        data["meta"] = meta
        return data

    # Signals collection methods
    def create_signal(self, signal_data: Dict[str, Any]) -> str:
        """Create a new signal document"""
        try:
            self._check_connection()
            signal_id = str(uuid.uuid4())
            signal_data["signal_id"] = signal_id
            
            # Add meta information
            signal_data = self._add_meta(signal_data)
            
            doc_ref = self.db.collection("signals").document(signal_id)
            doc_ref.set(signal_data)
            
            logger.info(f"Created signal: {signal_id} for symbol: {signal_data.get('symbol')}")
            return signal_id
        except Exception as e:
            logger.error(f"Failed to create signal: {e}")
            raise

    def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get a signal by ID"""
        try:
            doc_ref = self.db.collection("signals").document(signal_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get signal {signal_id}: {e}")
            raise

    def list_signals(self, status: Optional[str] = None, limit: int = 50, cursor: Optional[str] = None) -> Dict[str, Any]:
        """List signals with optional filtering - using simple query to avoid index issues"""
        try:
            # Get all signals first (no ordering to avoid index requirement)
            docs = self.db.collection("signals").stream()
            
            signals = []
            for doc in docs:
                signal_data = doc.to_dict()
                
                # Apply status filter in memory if needed
                if status and status != "all":
                    if signal_data.get("status") != status:
                        continue
                
                signals.append(signal_data)
            
            # Sort in memory by created_at (most recent first)
            signals.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # Apply limit
            signals = signals[:limit]
            
            return {
                "signals": signals,
                "total": len(signals),
                "cursor": None  # Simplified - no cursor for now
            }
        except Exception as e:
            logger.error(f"Failed to list signals: {e}")
            raise

    def update_signal_status(self, signal_id: str, status: str) -> bool:
        """Update signal status"""
        try:
            doc_ref = self.db.collection("signals").document(signal_id)
            doc_ref.update({"status": status})
            logger.info(f"Updated signal {signal_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}")
            return False

    # Positions collection methods
    def create_position(self, position_data: Dict[str, Any]) -> str:
        """Create a new position document"""
        try:
            position_id = str(uuid.uuid4())
            position_data["position_id"] = position_id
            
            # Add meta information
            position_data = self._add_meta(position_data)
            
            doc_ref = self.db.collection("positions").document(position_id)
            doc_ref.set(position_data)
            
            logger.info(f"Created position: {position_id} for symbol: {position_data.get('symbol')}")
            return position_id
        except Exception as e:
            logger.error(f"Failed to create position: {e}")
            raise

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a position by ID"""
        try:
            doc_ref = self.db.collection("positions").document(position_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}")
            raise

    def list_positions(self, status: str = "open") -> List[Dict[str, Any]]:
        """List positions by status"""
        try:
            query = self.db.collection("positions")
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            
            query = query.order_by("entered_at", direction=firestore.Query.DESCENDING)
            docs = query.stream()
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            raise

    def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """Update a position with new data"""
        try:
            self._check_connection()
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("positions").document(position_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated position: {position_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update position {position_id}: {e}")
            return False

    # Job collection methods
    def create_job(self, job_data: Dict[str, Any]) -> str:
        """Create a new analysis job"""
        try:
            self._check_connection()
            job_id = str(uuid.uuid4())
            job_data["job_id"] = job_id
            
            # Add meta information
            job_data = self._add_meta(job_data)
            
            doc_ref = self.db.collection("jobs").document(job_id)
            doc_ref.set(job_data)
            
            logger.info(f"Created job: {job_id} for symbol: {job_data.get('symbol')}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("jobs").document(job_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update a job with new data"""
        try:
            self._check_connection()
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("jobs").document(job_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False

    def update_job_stage(self, job_id: str, stage_name: str, stage_data: Dict[str, Any]) -> bool:
        """Update a specific stage within a job"""
        try:
            self._check_connection()
            
            # Add updated_at timestamp to stage data
            stage_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("jobs").document(job_id)
            doc_ref.update({
                f"stages.{stage_name}": stage_data,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Updated job {job_id} stage: {stage_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id} stage {stage_name}: {e}")
            return False

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List jobs with optional status filter"""
        try:
            self._check_connection()
            query = self.db.collection("jobs")
            
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            
            query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    def delete_job(self, job_id: str) -> bool:
        """Delete a job (for cleanup)"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("jobs").document(job_id)
            doc_ref.delete()
            
            logger.info(f"Deleted job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    # Symbol-based analysis retrieval methods
    def get_latest_analysis_by_symbol(self, symbol: str, analysis_type: str = "enhanced") -> Optional[Dict[str, Any]]:
        """Get most recent completed analysis for a symbol with complete data"""
        try:
            self._check_connection()
            query = (self.db.collection("jobs")
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("analysis_type", "==", analysis_type))
                    .where(filter=FieldFilter("status", "==", "completed"))
                    .order_by("created_at", direction=firestore.Query.DESCENDING))
            
            docs = list(query.stream())
            if not docs:
                logger.info(f"No completed analysis found for {symbol} (type: {analysis_type})")
                return None
            
            # Find the analysis with the most complete data
            best_analysis = None
            best_data_score = 0
            
            for doc in docs:
                analysis = doc.to_dict()
                stages = analysis.get("stages", {})
                
                # Calculate data completeness score
                data_score = 0
                for stage_name, stage_data in stages.items():
                    stage_data_content = stage_data.get("data", {})
                    if stage_data_content:  # If stage has data
                        data_score += len(stage_data_content)
                
                logger.info(f"Analysis {analysis.get('created_at')} has data score: {data_score}")
                
                if data_score > best_data_score:
                    best_data_score = data_score
                    best_analysis = analysis
            
            if best_analysis:
                logger.info(f"Found best analysis for {symbol} (type: {analysis_type}) - created: {best_analysis.get('created_at')}, data score: {best_data_score}")
                return best_analysis
            else:
                logger.info(f"No analysis with data found for {symbol} (type: {analysis_type})")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get latest analysis for {symbol}: {e}")
            return None

    def get_analysis_by_symbol_and_timeframe(self, symbol: str, days_back: int = 7, analysis_type: str = "enhanced") -> List[Dict[str, Any]]:
        """Get analysis for symbol within timeframe"""
        try:
            self._check_connection()
            from datetime import timedelta
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Get all completed analyses for the symbol first
            query = (self.db.collection("jobs")
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("analysis_type", "==", analysis_type))
                    .where(filter=FieldFilter("status", "==", "completed"))
                    .order_by("created_at", direction=firestore.Query.DESCENDING))
            
            docs = list(query.stream())
            analyses = []
            
            # Filter by date manually to handle different date formats
            for doc in docs:
                analysis = doc.to_dict()
                created_at_str = analysis.get("created_at", "")
                
                try:
                    # Try to parse the created_at string
                    if "T" in created_at_str and "Z" in created_at_str:
                        # ISO format with Z
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    elif "T" in created_at_str:
                        # ISO format without Z
                        created_at = datetime.fromisoformat(created_at_str)
                    else:
                        # Fallback to string comparison
                        created_at = datetime.fromisoformat(created_at_str)
                    
                    # Check if within timeframe
                    if created_at >= cutoff_date:
                        analyses.append(analysis)
                        
                except Exception as parse_error:
                    logger.warning(f"Could not parse date {created_at_str}: {parse_error}")
                    # If we can't parse the date, include it anyway (better to include than exclude)
                    analyses.append(analysis)
            
            logger.info(f"Found {len(analyses)} analyses for {symbol} in last {days_back} days")
            return analyses
        except Exception as e:
            logger.error(f"Failed to get analysis for {symbol} in last {days_back} days: {e}")
            return []

    def get_all_analyses_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all completed analyses for a symbol (for historical tracking)"""
        try:
            self._check_connection()
            query = (self.db.collection("jobs")
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("status", "==", "completed"))
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(limit))
            
            docs = list(query.stream())
            analyses = [doc.to_dict() for doc in docs]
            logger.info(f"Found {len(analyses)} total analyses for {symbol}")
            return analyses
        except Exception as e:
            logger.error(f"Failed to get all analyses for {symbol}: {e}")
            return []

    def is_analysis_fresh_for_investment(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if analysis is fresh enough for investment decisions"""
        try:
            created_at = datetime.fromisoformat(analysis_data["created_at"].replace('Z', '+00:00'))
            age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
            
            # Investment-focused freshness rules (longer cache times for wealth creation)
            freshness_rules = {
                "data_collection_and_analysis": 24 * 7,      # 7 days
                "technical_and_combined_scoring": 24 * 7,    # 7 days
                "forensic_analysis": 24 * 14,                # 14 days
                "module_selection": 24 * 14,                 # 14 days
                "risk_assessment": 24 * 3,                   # 3 days
                "final_decision": 24 * 1,                    # 1 day
                "verdict_synthesis": 24 * 7,                 # 7 days
                "final_scoring": 24 * 7                      # 7 days
            }
            
            # Check if any stage is too old
            stages = analysis_data.get("stages", {})
            stale_stages = []
            fresh_stages = []
            
            for stage_name, stage_data in stages.items():
                if stage_data.get("status") == "completed":
                    stage_age_hours = age_hours
                    max_age = freshness_rules.get(stage_name, 24 * 7)  # Default 7 days
                    
                    if stage_age_hours > max_age:
                        stale_stages.append({
                            "stage": stage_name,
                            "age_hours": stage_age_hours,
                            "max_age_hours": max_age,
                            "age_days": round(stage_age_hours / 24, 1)
                        })
                    else:
                        fresh_stages.append({
                            "stage": stage_name,
                            "age_hours": stage_age_hours,
                            "age_days": round(stage_age_hours / 24, 1)
                        })
            
            # Overall freshness decision
            is_fresh = len(stale_stages) == 0
            freshness_score = len(fresh_stages) / len(stages) if stages else 0
            
            return {
                "is_fresh": is_fresh,
                "age_hours": age_hours,
                "age_days": round(age_hours / 24, 1),
                "freshness_score": round(freshness_score, 2),
                "stale_stages": stale_stages,
                "fresh_stages": fresh_stages,
                "total_stages": len(stages),
                "recommendation": "refresh" if stale_stages else "use_cached"
            }
            
        except Exception as e:
            logger.error(f"Error checking analysis freshness: {e}")
            return {
                "is_fresh": False, 
                "error": str(e),
                "recommendation": "error"
            }

    # Fills collection methods
    def create_fill(self, fill_data: Dict[str, Any]) -> str:
        """Create a new fill document"""
        try:
            fill_id = str(uuid.uuid4())
            fill_data["fill_id"] = fill_id
            
            # Add meta information
            fill_data = self._add_meta(fill_data)
            
            doc_ref = self.db.collection("fills").document(fill_id)
            doc_ref.set(fill_data)
            
            logger.info(f"Created fill: {fill_id}")
            return fill_id
        except Exception as e:
            logger.error(f"Failed to create fill: {e}")
            raise

    # Config collection methods
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration"""
        try:
            doc_ref = self.db.collection("configs").document("runtime")
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                # Return default config if not exists
                default_config = {
                    "paper_mode": True,
                    "kill_switch": False,
                    "max_order_notional_minor": 2000000
                }
                self.set_runtime_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to get runtime config: {e}")
            raise

    def set_runtime_config(self, config: Dict[str, Any]) -> bool:
        """Set runtime configuration"""
        try:
            doc_ref = self.db.collection("configs").document("runtime")
            doc_ref.set(config)
            logger.info("Updated runtime config")
            return True
        except Exception as e:
            logger.error(f"Failed to set runtime config: {e}")
            return False

    # Audits collection methods
    def create_audit_log(self, action: str, details: Dict[str, Any], source: str = "system") -> str:
        """Create an audit log entry"""
        try:
            audit_id = str(uuid.uuid4())
            audit_data = {
                "id": audit_id,
                "action": action,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source
            }
            
            doc_ref = self.db.collection("audits").document(audit_id)
            doc_ref.set(audit_data)
            
            logger.info(f"Created audit log: {action}")
            return audit_id
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            raise

    # Utility methods
    def check_signal_exists_today(self, symbol: str, date_str: str) -> bool:
        """Check if signal already exists for symbol on given date (idempotency)"""
        try:
            query = (self.db.collection("signals")
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("created_at", ">=", f"{date_str}T00:00:00Z"))
                    .where(filter=FieldFilter("created_at", "<", f"{date_str}T23:59:59Z"))
                    .limit(1))
            
            docs = list(query.stream())
            return len(docs) > 0
        except Exception as e:
            logger.error(f"Failed to check signal exists: {e}")
            return False

# Singleton instance
firestore_client = FirestoreClient()
