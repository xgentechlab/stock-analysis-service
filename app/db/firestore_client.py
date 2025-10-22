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

    # Multi-timeframe Analysis collection methods
    def create_multi_timeframe_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Create a new multi-timeframe analysis document"""
        try:
            self._check_connection()
            analysis_id = analysis_data.get("analysis_id", str(uuid.uuid4()))
            
            doc_ref = self.db.collection("multi_timeframe_analyses").document(analysis_id)
            doc_ref.set(analysis_data)
            
            logger.info(f"Created multi-timeframe analysis {analysis_id} for {analysis_data.get('symbol', 'unknown')}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error creating multi-timeframe analysis: {e}")
            raise
    
    def get_multi_timeframe_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a multi-timeframe analysis by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("multi_timeframe_analyses").document(analysis_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe analysis {analysis_id}: {e}")
            return None
    
    def get_latest_multi_timeframe_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest multi-timeframe analysis for a symbol"""
        try:
            self._check_connection()
            query = (self.db.collection("multi_timeframe_analyses")
                    .where("symbol", "==", symbol)
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(1))
            
            docs = query.stream()
            for doc in docs:
                return doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest multi-timeframe analysis for {symbol}: {e}")
            return None
    
    def list_multi_timeframe_analyses(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List multi-timeframe analyses with optional symbol filter"""
        try:
            self._check_connection()
            query = self.db.collection("multi_timeframe_analyses")
            
            if symbol:
                query = query.where("symbol", "==", symbol)
            
            query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
            
            analyses = []
            for doc in query.stream():
                analyses.append(doc.to_dict())
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error listing multi-timeframe analyses: {e}")
            return []

    # Hot Stock Analysis collection methods
    def create_hot_stock_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Create a new hot stock analysis document"""
        try:
            self._check_connection()
            analysis_id = analysis_data.get("analysis_id", str(uuid.uuid4()))
            
            doc_ref = self.db.collection("hot_stock_analyses").document(analysis_id)
            doc_ref.set(analysis_data)
            
            logger.info(f"Created hot stock analysis {analysis_id} for {analysis_data.get('symbol', 'unknown')}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error creating hot stock analysis: {e}")
            raise
    
    def get_hot_stock_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a hot stock analysis by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("hot_stock_analyses").document(analysis_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting hot stock analysis {analysis_id}: {e}")
            return None
    
    def get_latest_hot_stock_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest hot stock analysis for a symbol"""
        try:
            self._check_connection()
            query = (self.db.collection("hot_stock_analyses")
                    .where("symbol", "==", symbol)
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(1))
            
            docs = query.stream()
            for doc in docs:
                return doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest hot stock analysis for {symbol}: {e}")
            return None
    
    def list_hot_stock_analyses(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List hot stock analyses with optional symbol filter"""
        try:
            self._check_connection()
            query = self.db.collection("hot_stock_analyses")
            
            if symbol:
                query = query.where("symbol", "==", symbol)
            
            query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
            
            analyses = []
            for doc in query.stream():
                analyses.append(doc.to_dict())
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error listing hot stock analyses: {e}")
            return []

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
    def get_job_analysis_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get complete analysis data for a specific job ID"""
        try:
            self._check_connection()
            logger.info(f"🔍 FIRESTORE QUERY: Fetching analysis data for job {job_id}")
            
            doc_ref = self.db.collection("jobs").document(job_id)
            doc = doc_ref.get()
            
            if doc.exists:
                job_data = doc.to_dict()
                logger.info(f"📊 FIRESTORE RESULT: Found job data for {job_id}")
                return job_data
            else:
                logger.warning(f"📊 FIRESTORE RESULT: Job {job_id} not found")
                return None
        except Exception as e:
            logger.error(f"Failed to get job analysis data for {job_id}: {e}")
            return None

    def get_latest_analysis_by_symbol(self, symbol: str, analysis_type: str = "enhanced") -> Optional[Dict[str, Any]]:
        """Get most recent completed analysis for a symbol with complete data"""
        try:
            self._check_connection()
            logger.info(f"🔍 FIRESTORE QUERY: Looking for completed analysis for {symbol} (type: {analysis_type})")
            
            query = (self.db.collection("jobs")
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("analysis_type", "==", analysis_type))
                    .where(filter=FieldFilter("status", "==", "completed"))
                    .order_by("created_at", direction=firestore.Query.DESCENDING))
            
            logger.info(f"📊 FIRESTORE QUERY: Executing query for {symbol}")
            docs = list(query.stream())
            logger.info(f"📊 FIRESTORE RESULT: Found {len(docs)} completed jobs for {symbol}")
            
            if not docs:
                logger.warning(f"⚠️  FIRESTORE: No completed analysis found for {symbol} (type: {analysis_type})")
                
                # Let's check what statuses actually exist for this symbol
                logger.info(f"🔍 DEBUGGING: Checking all statuses for {symbol}")
                all_statuses_query = (self.db.collection("jobs")
                                    .where(filter=FieldFilter("symbol", "==", symbol))
                                    .where(filter=FieldFilter("analysis_type", "==", analysis_type)))
                
                all_statuses_docs = list(all_statuses_query.stream())
                logger.info(f"📊 DEBUGGING: Found {len(all_statuses_docs)} total jobs for {symbol} (type: {analysis_type})")
                
                statuses = {}
                for doc in all_statuses_docs:
                    job_data = doc.to_dict()
                    status = job_data.get('status', 'unknown')
                    statuses[status] = statuses.get(status, 0) + 1
                
                logger.info(f"📊 DEBUGGING: Status breakdown for {symbol}: {statuses}")
                return None
            
            # Find the analysis with the most complete data
            best_analysis = None
            best_data_score = 0
            
            logger.info(f"🔍 FIRESTORE: Analyzing {len(docs)} completed jobs for {symbol}")
            
            for i, doc in enumerate(docs):
                analysis = doc.to_dict()
                stages = analysis.get("stages", {})
                job_id = analysis.get('job_id', 'unknown')
                created_at = analysis.get('created_at', 'unknown')
                
                # Calculate data completeness score
                data_score = 0
                completed_stages = 0
                for stage_name, stage_data in stages.items():
                    stage_status = stage_data.get('status', 'unknown')
                    stage_data_content = stage_data.get("data", {})
                    if stage_data_content:  # If stage has data
                        data_score += len(stage_data_content)
                    if stage_status == 'completed':
                        completed_stages += 1
                
                logger.info(f"  Job {i+1}: ID={job_id}, Created={created_at}, Data Score={data_score}, Completed Stages={completed_stages}")
                
                if data_score > best_data_score:
                    best_data_score = data_score
                    best_analysis = analysis
                    logger.info(f"  ✅ NEW BEST: Job {job_id} is now the best analysis (score: {data_score})")
            
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

    # ===== USER MANAGEMENT COLLECTIONS =====
    
    # Recommendations collection methods
    def create_recommendation(self, recommendation_data: Dict[str, Any]) -> str:
        """Create or update recommendation with deduplication"""
        try:
            self._check_connection()
            user_id = recommendation_data.get('user_id')
            symbol = recommendation_data.get('symbol')
            action = recommendation_data.get('action')
            
            # Check for existing recommendation for same user+symbol+action
            existing = self.get_existing_recommendation(user_id, symbol, action)
            
            if existing:
                # Update existing recommendation with new data
                recommendation_id = existing['id']
                recommendation_data["id"] = recommendation_id
                recommendation_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                doc_ref = self.db.collection("recommendations").document(recommendation_id)
                doc_ref.update(recommendation_data)
                
                logger.info(f"Updated existing recommendation: {recommendation_id} for user: {user_id}, symbol: {symbol}, action: {action}")
                return recommendation_id
            else:
                # Create new recommendation
                recommendation_id = str(uuid.uuid4())
                recommendation_data["id"] = recommendation_id
                recommendation_data["created_at"] = datetime.now(timezone.utc).isoformat()
                
                # Set default status if not provided
                if "status" not in recommendation_data:
                    recommendation_data["status"] = "pending"
                
                doc_ref = self.db.collection("recommendations").document(recommendation_id)
                doc_ref.set(recommendation_data)
                
                logger.info(f"Created new recommendation: {recommendation_id} for user: {user_id}, symbol: {symbol}, action: {action}")
                return recommendation_id
        except Exception as e:
            logger.error(f"Failed to create/update recommendation: {e}")
            raise

    def get_recommendation(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """Get a recommendation by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("recommendations").document(recommendation_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get recommendation {recommendation_id}: {e}")
            raise

    def get_existing_recommendation(self, user_id: str, symbol: str, action: str) -> Optional[Dict[str, Any]]:
        """Get existing recommendation for user+symbol+action combination"""
        try:
            self._check_connection()
            # Query without status filter first, then filter in memory
            query = (self.db.collection("recommendations")
                    .where(filter=FieldFilter("user_id", "==", user_id))
                    .where(filter=FieldFilter("symbol", "==", symbol))
                    .where(filter=FieldFilter("action", "==", action))
                    .limit(10))  # Get more to filter in memory
            
            docs = list(query.stream())
            
            # Filter for pending status or no status (defaults to pending)
            for doc in docs:
                data = doc.to_dict()
                status = data.get("status", "pending")  # Default to pending if not set
                if status == "pending":
                    return data
            
            return None
        except Exception as e:
            logger.error(f"Failed to check existing recommendation: {e}")
            return None

    def update_recommendation(self, recommendation_id: str, updates: Dict[str, Any]) -> bool:
        """Update a recommendation"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("recommendations").document(recommendation_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated recommendation: {recommendation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update recommendation {recommendation_id}: {e}")
            return False

    def list_recommendations(self, user_id: Optional[str] = None, status: Optional[str] = None, 
                           limit: int = 50, cursor: Optional[str] = None, exclude_acted: bool = True) -> Dict[str, Any]:
        """List recommendations with optional filtering"""
        try:
            self._check_connection()
            query = self.db.collection("recommendations")
            if user_id:
                query = query.where(filter=FieldFilter("user_id", "==", user_id))
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            # Avoid composite index requirement: stream then sort in memory
            docs = list(query.stream())
            recommendations = [doc.to_dict() for doc in docs]
            
            # Filter out recommendations that have been acted upon
            if exclude_acted:
                acted_upon_ids = set()
                if user_id:
                    # Get all decisions for this user
                    decisions = self.list_user_decisions(user_id=user_id, limit=1000)
                    acted_upon_ids = {decision.get("recommendation_id") for decision in decisions}
                
                # Filter out recommendations that have been acted upon
                recommendations = [rec for rec in recommendations if rec.get("id") not in acted_upon_ids]
            
            recommendations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            recommendations = recommendations[:limit]
            
            return {
                "recommendations": recommendations,
                "total": len(recommendations),
                "cursor": None  # Simplified - no cursor for now
            }
        except Exception as e:
            logger.error(f"Failed to list recommendations: {e}")
            raise

    def delete_recommendation(self, recommendation_id: str) -> bool:
        """Delete a recommendation"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("recommendations").document(recommendation_id)
            doc_ref.delete()
            
            logger.info(f"Deleted recommendation: {recommendation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete recommendation {recommendation_id}: {e}")
            return False

    # Watchlist collection methods
    def add_to_watchlist(self, watchlist_data: Dict[str, Any]) -> str:
        """Add a symbol to user's watchlist with duplicate prevention"""
        try:
            self._check_connection()
            user_id = watchlist_data['user_id']
            symbol = watchlist_data['symbol']
            
            # Check if already exists
            if self.is_in_watchlist(user_id, symbol):
                logger.info(f"Symbol {symbol} already in watchlist for user {user_id} - skipping duplicate")
                return f"{user_id}_{symbol}"
            
            # Add to watchlist
            watchlist_data["added_at"] = datetime.now(timezone.utc).isoformat()
            
            # Use composite key: user_id + symbol
            doc_id = f"{user_id}_{symbol}"
            doc_ref = self.db.collection("watchlist").document(doc_id)
            doc_ref.set(watchlist_data)
            
            logger.info(f"Added {symbol} to watchlist for user: {user_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add to watchlist: {e}")
            raise

    def remove_from_watchlist(self, user_id: str, symbol: str) -> bool:
        """Remove a symbol from user's watchlist"""
        try:
            self._check_connection()
            doc_id = f"{user_id}_{symbol}"
            doc_ref = self.db.collection("watchlist").document(doc_id)
            doc_ref.delete()
            
            logger.info(f"Removed {symbol} from watchlist for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove from watchlist: {e}")
            return False

    def get_user_watchlist(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's watchlist"""
        try:
            self._check_connection()
            query = self.db.collection("watchlist").where(filter=FieldFilter("user_id", "==", user_id))
            docs = list(query.stream())
            items = [doc.to_dict() for doc in docs]
            # Sort in memory by added_at desc to avoid requiring composite index
            items.sort(key=lambda x: x.get("added_at", ""), reverse=True)
            return items
        except Exception as e:
            logger.error(f"Failed to get watchlist for user {user_id}: {e}")
            raise

    def is_in_watchlist(self, user_id: str, symbol: str) -> bool:
        """Check if symbol is in user's watchlist"""
        try:
            self._check_connection()
            doc_id = f"{user_id}_{symbol}"
            doc_ref = self.db.collection("watchlist").document(doc_id)
            doc = doc_ref.get()
            
            return doc.exists
        except Exception as e:
            logger.error(f"Failed to check watchlist: {e}")
            return False

    # Portfolio collection methods
    def create_portfolio_item(self, portfolio_data: Dict[str, Any]) -> str:
        """Create a new portfolio item"""
        try:
            self._check_connection()
            portfolio_id = str(uuid.uuid4())
            portfolio_data["id"] = portfolio_id
            
            if not portfolio_data.get("entry_date"):
                portfolio_data["entry_date"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("portfolio").document(portfolio_id)
            doc_ref.set(portfolio_data)
            
            logger.info(f"Created portfolio item: {portfolio_id} for user: {portfolio_data.get('user_id')}")
            return portfolio_id
        except Exception as e:
            logger.error(f"Failed to create portfolio item: {e}")
            raise

    def get_portfolio_item(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get a portfolio item by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio").document(portfolio_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get portfolio item {portfolio_id}: {e}")
            raise

    def update_portfolio_item(self, portfolio_id: str, updates: Dict[str, Any]) -> bool:
        """Update a portfolio item"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio").document(portfolio_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated portfolio item: {portfolio_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update portfolio item {portfolio_id}: {e}")
            return False

    def get_user_portfolio(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user's portfolio"""
        try:
            self._check_connection()
            query = self.db.collection("portfolio").where(filter=FieldFilter("user_id", "==", user_id))
            
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            docs = list(query.stream())
            items = [doc.to_dict() for doc in docs]
            items.sort(key=lambda x: x.get("entry_date", ""), reverse=True)
            return items
        except Exception as e:
            logger.error(f"Failed to get portfolio for user {user_id}: {e}")
            raise

    def delete_portfolio_item(self, portfolio_id: str) -> bool:
        """Delete a portfolio item"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio").document(portfolio_id)
            doc_ref.delete()
            
            logger.info(f"Deleted portfolio item: {portfolio_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete portfolio item {portfolio_id}: {e}")
            return False

    # User Decisions collection methods
    def create_user_decision(self, decision_data: Dict[str, Any]) -> str:
        """Create a new user decision"""
        try:
            self._check_connection()
            decision_id = str(uuid.uuid4())
            decision_data["id"] = decision_id
            decision_data["decided_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("user_decisions").document(decision_id)
            doc_ref.set(decision_data)
            
            logger.info(f"Created user decision: {decision_id}")
            return decision_id
        except Exception as e:
            logger.error(f"Failed to create user decision: {e}")
            raise

    def get_user_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a user decision by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("user_decisions").document(decision_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get user decision {decision_id}: {e}")
            raise

    def get_decisions_for_recommendation(self, recommendation_id: str) -> List[Dict[str, Any]]:
        """Get all decisions for a specific recommendation"""
        try:
            self._check_connection()
            query = self.db.collection("user_decisions").where(filter=FieldFilter("recommendation_id", "==", recommendation_id))
            docs = list(query.stream())
            items = [doc.to_dict() for doc in docs]
            items.sort(key=lambda x: x.get("decided_at", ""), reverse=True)
            return items
        except Exception as e:
            logger.error(f"Failed to get decisions for recommendation {recommendation_id}: {e}")
            raise

    def list_user_decisions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List user decisions with optional filtering"""
        try:
            self._check_connection()
            query = self.db.collection("user_decisions")
            
            if user_id:
                query = query.where(filter=FieldFilter("user_id", "==", user_id))
            
            # Avoid composite index requirement: stream then sort in memory
            docs = list(query.stream())
            decisions = [doc.to_dict() for doc in docs]
            
            # Sort by decided_at descending and limit
            decisions.sort(key=lambda x: x.get("decided_at", ""), reverse=True)
            decisions = decisions[:limit]
            
            return decisions
        except Exception as e:
            logger.error(f"Failed to list user decisions: {e}")
            raise

    # Portfolio Suggestions collection methods
    def create_portfolio_suggestion(self, suggestion_data: Dict[str, Any]) -> str:
        """Create a new portfolio suggestion"""
        try:
            self._check_connection()
            suggestion_id = str(uuid.uuid4())
            suggestion_data["id"] = suggestion_id
            suggestion_data["created_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("portfolio_suggestions").document(suggestion_id)
            doc_ref.set(suggestion_data)
            
            logger.info(f"Created portfolio suggestion: {suggestion_id} for user: {suggestion_data.get('user_id')}")
            return suggestion_id
        except Exception as e:
            logger.error(f"Failed to create portfolio suggestion: {e}")
            raise

    def get_portfolio_suggestion(self, suggestion_id: str) -> Optional[Dict[str, Any]]:
        """Get a portfolio suggestion by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio_suggestions").document(suggestion_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get portfolio suggestion {suggestion_id}: {e}")
            raise

    def update_portfolio_suggestion(self, suggestion_id: str, updates: Dict[str, Any]) -> bool:
        """Update a portfolio suggestion"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio_suggestions").document(suggestion_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated portfolio suggestion: {suggestion_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update portfolio suggestion {suggestion_id}: {e}")
            return False

    def get_user_portfolio_suggestions(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get user's portfolio suggestions"""
        try:
            self._check_connection()
            query = self.db.collection("portfolio_suggestions").where(filter=FieldFilter("user_id", "==", user_id))
            
            if status:
                query = query.where(filter=FieldFilter("status", "==", status))
            docs = list(query.stream())
            items = [doc.to_dict() for doc in docs]
            items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return items
        except Exception as e:
            logger.error(f"Failed to get portfolio suggestions for user {user_id}: {e}")
            raise

    def delete_portfolio_suggestion(self, suggestion_id: str) -> bool:
        """Delete a portfolio suggestion"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("portfolio_suggestions").document(suggestion_id)
            doc_ref.delete()
            
            logger.info(f"Deleted portfolio suggestion: {suggestion_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete portfolio suggestion {suggestion_id}: {e}")
            return False

    # ===== STOCKS COLLECTION METHODS =====
    
    def create_stock(self, stock_data: Dict[str, Any]) -> str:
        """Create a new stock"""
        try:
            self._check_connection()
            stock_id = str(uuid.uuid4())
            stock_data["id"] = stock_id
            stock_data["created_at"] = datetime.now(timezone.utc).isoformat()
            stock_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Set default values
            if "is_active" not in stock_data:
                stock_data["is_active"] = True
            
            doc_ref = self.db.collection("stocks").document(stock_id)
            doc_ref.set(stock_data)
            
            logger.info(f"Created stock: {stock_id} for symbol: {stock_data.get('symbol')}")
            return stock_id
        except Exception as e:
            logger.error(f"Failed to create stock: {e}")
            raise

    def get_stock(self, stock_id: str) -> Optional[Dict[str, Any]]:
        """Get a stock by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("stocks").document(stock_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get stock {stock_id}: {e}")
            raise

    def get_stock_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a stock by symbol"""
        try:
            self._check_connection()
            query = self.db.collection("stocks").where(filter=FieldFilter("symbol", "==", symbol))
            docs = list(query.stream())
            
            if docs:
                return docs[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get stock by symbol {symbol}: {e}")
            raise

    def update_stock(self, stock_id: str, updates: Dict[str, Any]) -> bool:
        """Update a stock"""
        try:
            self._check_connection()
            updates["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            doc_ref = self.db.collection("stocks").document(stock_id)
            doc_ref.update(updates)
            
            logger.info(f"Updated stock: {stock_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update stock {stock_id}: {e}")
            return False

    def delete_stock(self, stock_id: str) -> bool:
        """Delete a stock (soft delete by setting is_active=False)"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("stocks").document(stock_id)
            doc_ref.update({
                "is_active": False,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Soft deleted stock: {stock_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete stock {stock_id}: {e}")
            return False

    def list_stocks(self, industry: Optional[str] = None, is_active: bool = True, 
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List stocks with optional filtering"""
        try:
            self._check_connection()
            query = self.db.collection("stocks")
            
            # Apply filters
            if is_active is not None:
                query = query.where(filter=FieldFilter("is_active", "==", is_active))
            
            if industry:
                query = query.where(filter=FieldFilter("industry", "==", industry))
            
            # Get all documents and sort in memory to avoid index requirements
            docs = list(query.stream())
            stocks = [doc.to_dict() for doc in docs]
            
            # Sort by company_name
            stocks.sort(key=lambda x: x.get("company_name", ""))
            
            # Apply pagination
            total = len(stocks)
            start_idx = offset
            end_idx = offset + limit
            paginated_stocks = stocks[start_idx:end_idx]
            
            return {
                "stocks": paginated_stocks,
                "total": total,
                "page": (offset // limit) + 1,
                "limit": limit,
                "has_more": end_idx < total
            }
        except Exception as e:
            logger.error(f"Failed to list stocks: {e}")
            raise

    def search_stocks(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search stocks by company name or symbol"""
        try:
            self._check_connection()
            # Get all active stocks
            query = self.db.collection("stocks").where(filter=FieldFilter("is_active", "==", True))
            docs = list(query.stream())
            stocks = [doc.to_dict() for doc in docs]
            
            # Filter in memory (case-insensitive search)
            search_term_lower = search_term.lower()
            filtered_stocks = []
            
            for stock in stocks:
                company_name = stock.get("company_name", "").lower()
                symbol = stock.get("symbol", "").lower()
                
                if (search_term_lower in company_name or 
                    search_term_lower in symbol):
                    filtered_stocks.append(stock)
            
            # Sort by relevance (exact symbol match first, then company name)
            filtered_stocks.sort(key=lambda x: (
                0 if x.get("symbol", "").lower() == search_term_lower else
                1 if search_term_lower in x.get("symbol", "").lower() else
                2
            ))
            
            return filtered_stocks[:limit]
        except Exception as e:
            logger.error(f"Failed to search stocks: {e}")
            raise

    def bulk_create_stocks(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk create stocks (for initial data population)"""
        try:
            self._check_connection()
            created_count = 0
            failed_count = 0
            errors = []
            
            # Use batch writes for efficiency
            batch = self.db.batch()
            batch_size = 500  # Firestore batch limit
            
            for i, stock_data in enumerate(stocks_data):
                try:
                    stock_id = str(uuid.uuid4())
                    stock_data["id"] = stock_id
                    stock_data["created_at"] = datetime.now(timezone.utc).isoformat()
                    stock_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                    
                    if "is_active" not in stock_data:
                        stock_data["is_active"] = True
                    
                    doc_ref = self.db.collection("stocks").document(stock_id)
                    batch.set(doc_ref, stock_data)
                    
                    # Commit batch when it reaches the limit
                    if (i + 1) % batch_size == 0:
                        batch.commit()
                        batch = self.db.batch()
                        logger.info(f"Committed batch: {i + 1} stocks processed")
                    
                    created_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    errors.append(f"Stock {stock_data.get('symbol', 'unknown')}: {str(e)}")
                    logger.error(f"Failed to prepare stock {stock_data.get('symbol')}: {e}")
            
            # Commit remaining batch
            if created_count % batch_size != 0:
                batch.commit()
            
            logger.info(f"Bulk create completed: {created_count} created, {failed_count} failed")
            
            return {
                "created_count": created_count,
                "failed_count": failed_count,
                "errors": errors[:10]  # Limit error details
            }
        except Exception as e:
            logger.error(f"Failed to bulk create stocks: {e}")
            raise

    def get_stocks_by_industry(self, industry: str) -> List[Dict[str, Any]]:
        """Get all stocks in a specific industry"""
        try:
            self._check_connection()
            query = (self.db.collection("stocks")
                    .where(filter=FieldFilter("industry", "==", industry))
                    .where(filter=FieldFilter("is_active", "==", True)))
            
            docs = list(query.stream())
            stocks = [doc.to_dict() for doc in docs]
            
            # Sort by company name
            stocks.sort(key=lambda x: x.get("company_name", ""))
            
            return stocks
        except Exception as e:
            logger.error(f"Failed to get stocks by industry {industry}: {e}")
            raise

    def get_all_industries(self) -> List[str]:
        """Get list of all unique industries"""
        try:
            self._check_connection()
            query = self.db.collection("stocks").where(filter=FieldFilter("is_active", "==", True))
            docs = list(query.stream())
            
            industries = set()
            for doc in docs:
                stock_data = doc.to_dict()
                industry = stock_data.get("industry")
                if industry:
                    industries.add(industry)
            
            return sorted(list(industries))
        except Exception as e:
            logger.error(f"Failed to get industries: {e}")
            raise

    # Hot Stocks Runs collection methods
    def create_hot_stocks_run(self, run_data: Dict[str, Any]) -> str:
        """Create a new hot stocks run document"""
        try:
            self._check_connection()
            run_id = str(uuid.uuid4())
            run_data["run_id"] = run_id
            
            # Add meta information
            run_data = self._add_meta(run_data)
            
            doc_ref = self.db.collection("hot_stocks_runs").document(run_id)
            doc_ref.set(run_data)
            
            logger.info(f"Created hot stocks run: {run_id} with {len(run_data.get('hot_stocks', []))} stocks")
            return run_id
        except Exception as e:
            logger.error(f"Failed to create hot stocks run: {e}")
            raise

    def get_hot_stocks_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a hot stocks run by ID"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("hot_stocks_runs").document(run_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get hot stocks run {run_id}: {e}")
            raise

    def get_latest_hot_stocks_run(self) -> Optional[Dict[str, Any]]:
        """Get the latest hot stocks run"""
        try:
            self._check_connection()
            # Try ordering by created_at first (more reliable)
            query = (self.db.collection("hot_stocks_runs")
                    .order_by("created_at", direction=firestore.Query.DESCENDING)
                    .limit(1))
            
            docs = list(query.stream())
            if docs:
                return docs[0].to_dict()
            
            # Fallback to run_timestamp if created_at doesn't work
            query = (self.db.collection("hot_stocks_runs")
                    .order_by("run_timestamp", direction=firestore.Query.DESCENDING)
                    .limit(1))
            
            docs = list(query.stream())
            if docs:
                return docs[0].to_dict()
            
            return None
        except Exception as e:
            logger.error(f"Failed to get latest hot stocks run: {e}")
            raise

    def list_hot_stocks_runs(self, limit: int = 50, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """List hot stocks runs with optional date filtering"""
        try:
            self._check_connection()
            query = self.db.collection("hot_stocks_runs").order_by("run_timestamp", direction=firestore.Query.DESCENDING)
            
            # Apply date filters if provided
            if start_date:
                query = query.where("run_timestamp", ">=", start_date)
            if end_date:
                query = query.where("run_timestamp", "<=", end_date)
            
            query = query.limit(limit)
            docs = list(query.stream())
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error listing hot stocks runs: {e}")
            return []

    def get_hot_stocks_runs_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get hot stocks runs within a date range"""
        try:
            self._check_connection()
            query = (self.db.collection("hot_stocks_runs")
                    .where("run_timestamp", ">=", start_date)
                    .where("run_timestamp", "<=", end_date)
                    .order_by("run_timestamp", direction=firestore.Query.DESCENDING))
            
            docs = list(query.stream())
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error getting hot stocks runs by date range: {e}")
            return []

    def delete_hot_stocks_run(self, run_id: str) -> bool:
        """Delete a hot stocks run"""
        try:
            self._check_connection()
            doc_ref = self.db.collection("hot_stocks_runs").document(run_id)
            doc_ref.delete()
            
            logger.info(f"Deleted hot stocks run: {run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete hot stocks run {run_id}: {e}")
            return False

# Singleton instance
firestore_client = FirestoreClient()
