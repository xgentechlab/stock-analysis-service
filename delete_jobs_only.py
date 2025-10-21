#!/usr/bin/env python3
"""
Simple script to delete only jobs data from Firestore database
This script will remove all documents from the 'jobs' collection only.

WARNING: This will permanently delete all job data!
"""

import os
import sys
import logging
from datetime import datetime, timezone

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db.firestore_client import firestore_client
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def delete_jobs_collection():
    """Delete all documents from the jobs collection"""
    try:
        # Check if Firestore is properly configured
        if not settings.firestore_project_id:
            logger.error("❌ Firestore project ID not configured!")
            return False
        
        if not firestore_client.db:
            logger.error("❌ Firestore client not initialized!")
            return False
        
        logger.info("🚀 Jobs Deletion Script Started")
        logger.info(f"📋 Target project: {settings.firestore_project_id}")
        logger.info(f"📋 Target database: {settings.firestore_database_id}")
        
        # Get current job count
        jobs_query = firestore_client.db.collection("jobs")
        jobs_docs = list(jobs_query.stream())
        job_count = len(jobs_docs)
        
        if job_count == 0:
            logger.info("✅ No jobs found to delete.")
            return True
        
        logger.info(f"📊 Found {job_count} jobs to delete")
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will permanently delete {job_count} jobs!")
        response = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
        
        if response not in ['yes', 'y']:
            logger.info("❌ Deletion cancelled by user")
            return False
        
        # Delete jobs in batches
        deleted_count = 0
        batch_size = 100
        
        logger.info("🗑️  Starting job deletion...")
        
        while True:
            # Get a batch of jobs
            jobs_batch = list(firestore_client.db.collection("jobs").limit(batch_size).stream())
            
            if not jobs_batch:
                break
            
            # Delete batch
            batch = firestore_client.db.batch()
            for job_doc in jobs_batch:
                batch.delete(job_doc.reference)
                deleted_count += 1
            
            # Commit batch
            batch.commit()
            logger.info(f"  Deleted batch of {len(jobs_batch)} jobs")
        
        # Verify deletion
        remaining_jobs = list(firestore_client.db.collection("jobs").stream())
        
        if len(remaining_jobs) == 0:
            logger.info(f"✅ Successfully deleted {deleted_count} jobs")
            
            # Create audit log
            try:
                firestore_client.create_audit_log(
                    action="jobs_cleanup",
                    details={
                        "jobs_deleted": deleted_count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "script": "delete_jobs_only.py"
                    },
                    source="cleanup_script"
                )
                logger.info("📝 Created audit log")
            except Exception as e:
                logger.warning(f"Failed to create audit log: {e}")
            
            return True
        else:
            logger.warning(f"⚠️  Deleted {deleted_count} jobs, but {len(remaining_jobs)} remain")
            return False
            
    except Exception as e:
        logger.error(f"❌ Deletion failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = delete_jobs_collection()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n❌ Deletion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
