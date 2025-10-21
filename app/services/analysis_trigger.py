"""
Reusable analysis trigger service to initiate stock analysis without HTTP.
Matches the behavior of the initiate_analysis route, including cache checks
and background processing via stage_processor.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Any, Optional, Tuple

from app.models.schemas import JobCreateRequest
from app.services.job_service import job_service
from app.services.stage_processor import stage_processor

logger = logging.getLogger(__name__)


class AnalysisTrigger:
    """Helper to trigger analysis jobs directly from Python code."""

    def trigger_analysis(self, request: JobCreateRequest, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Create an analysis job and start processing in a background thread.

        Returns a dict containing job_id, symbol, analysis_type, status, and cache_info
        similar to the /api/v1/analyze response.
        """
        try:
            logger.info(
                "[analysis_trigger] Triggering analysis | symbol=%s type=%s force_refresh=%s",
                request.symbol,
                request.analysis_type.value,
                force_refresh,
            )
            cache_info = {
                "cache_available": False,
                "cache_fresh": False,
                "cache_age_days": None,
                "cache_recommendation": "new_analysis",
                "cost_saved": 0.00,
            }

            if not force_refresh:
                try:
                    # Lazy import to avoid import cycles on app startup
                    from app.db.firestore_client import firestore_client

                    cached_analysis = firestore_client.get_latest_analysis_by_symbol(
                        request.symbol,
                        request.analysis_type.value,
                    )

                    if cached_analysis:
                        freshness_check = firestore_client.is_analysis_fresh_for_investment(cached_analysis)
                        cache_info.update({
                            "cache_available": True,
                            "cache_fresh": freshness_check.get("is_fresh", False),
                            "cache_age_days": freshness_check.get("age_days"),
                            "cache_recommendation": freshness_check.get("recommendation"),
                            "cost_saved": 0.10 if freshness_check.get("is_fresh") else 0.00,
                        })
                        logger.info(
                            "[analysis_trigger] Cache check | symbol=%s fresh=%s age_days=%s recommendation=%s",
                            request.symbol,
                            cache_info["cache_fresh"],
                            cache_info["cache_age_days"],
                            cache_info["cache_recommendation"],
                        )
                except Exception as cache_e:
                    logger.warning(f"[analysis_trigger] Cache check failed for {request.symbol}: {cache_e}")

            # Create job document
            job_id = job_service.create_job(request)
            logger.info(
                "[analysis_trigger] Job created | symbol=%s job_id=%s status=processing",
                request.symbol,
                job_id,
            )

            # Background processing
            def run_background_task():
                try:
                    logger.info("[analysis_trigger] Background processing start | job_id=%s", job_id)
                    success = stage_processor.process_job(job_id)
                    if not success:
                        logger.error(f"[analysis_trigger] Job processing failed | job_id={job_id}")
                    else:
                        logger.info("[analysis_trigger] Background processing finished | job_id=%s", job_id)
                except Exception as e:
                    logger.error(f"[analysis_trigger] Error processing job {job_id}: {e}")

            thread = threading.Thread(target=run_background_task)
            thread.daemon = True
            thread.start()
            logger.info(
                "[analysis_trigger] Background thread started | job_id=%s thread_ident=%s",
                job_id,
                thread.ident,
            )

            message: str
            if cache_info["cache_available"] and cache_info["cache_fresh"]:
                message = (
                    f"Analysis job created - will use cached data (saved ${cache_info['cost_saved']:.2f}, "
                    f"age: {cache_info['cache_age_days']:.1f} days)"
                )
            elif cache_info["cache_available"] and not cache_info["cache_fresh"]:
                message = (
                    f"Analysis job created - cached data is stale, creating fresh analysis (age: "
                    f"{cache_info['cache_age_days']:.1f} days)"
                )
            else:
                message = "Analysis job created - no cached data found, creating fresh analysis"

            return {
                "job_id": job_id,
                "symbol": request.symbol,
                "analysis_type": request.analysis_type.value,
                "status": "processing",
                "message": message,
                "cache_info": cache_info,
            }
        except Exception as e:
            logger.error(f"Failed to trigger analysis for {request.symbol}: {e}")
            raise

    def fire_and_forget(self, request: JobCreateRequest, force_refresh: bool = False) -> str:
        """
        Fire-and-forget initiation. Returns the job_id immediately.
        """
        result = self.trigger_analysis(request, force_refresh=force_refresh)
        return result["job_id"]


# Singleton instance
analysis_trigger = AnalysisTrigger()


