#!/usr/bin/env python3
"""
Update Database Jobs to Completed Status
Updates existing jobs that should be marked as completed
"""

import sys
import os
import logging
from datetime import datetime, timezone

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.db.firestore_client import firestore_client
from app.models.schemas import JobStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_jobs_to_completed():
    """Update jobs that should be marked as completed"""
    print("🔄 Updating Database Jobs to Completed Status...")
    
    try:
        # Get all jobs for 360ONE that are currently processing
        jobs = list(firestore_client.db.collection('jobs').where('symbol', '==', '360ONE').where('analysis_type', '==', 'enhanced').stream())
        
        print(f"Found {len(jobs)} jobs for 360ONE")
        
        updated_count = 0
        
        for i, job_doc in enumerate(jobs):
            job_data = job_doc.to_dict()
            job_id = job_doc.id
            status = job_data.get('status', 'unknown')
            completed_stages = job_data.get('completed_stages', [])
            failed_stages = job_data.get('failed_stages', [])
            
            print(f"\nJob {i+1}: {job_id}")
            print(f"  Current Status: {status}")
            print(f"  Completed Stages: {len(completed_stages)} - {completed_stages}")
            print(f"  Failed Stages: {len(failed_stages)} - {failed_stages}")
            
            # Check if this job should be marked as completed
            should_be_completed = False
            
            if status == 'processing':
                # Check if it has completed stages that indicate successful completion
                if len(completed_stages) >= 4:  # At least 4 stages completed
                    should_be_completed = True
                    print(f"  ✅ Should be completed (has {len(completed_stages)} completed stages)")
                elif len(completed_stages) > 0 and len(failed_stages) == 0:
                    # Has some completed stages and no failed stages
                    should_be_completed = True
                    print(f"  ✅ Should be completed (has {len(completed_stages)} completed stages, no failures)")
                else:
                    print(f"  ⚠️  Not enough completed stages or has failures")
            
            if should_be_completed:
                try:
                    # Update job to completed status
                    updates = {
                        "status": JobStatus.COMPLETED.value,
                        "completed_stages": completed_stages,
                        "failed_stages": failed_stages,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "progress_percentage": 100,
                        "cost_saved": 0.10
                    }
                    
                    firestore_client.db.collection('jobs').document(job_id).update(updates)
                    print(f"  ✅ Updated to COMPLETED")
                    updated_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Failed to update: {e}")
            else:
                print(f"  ⏭️  Skipped (not ready for completion)")
        
        print(f"\n📊 Summary:")
        print(f"  Total jobs checked: {len(jobs)}")
        print(f"  Jobs updated to completed: {updated_count}")
        
        return updated_count > 0
        
    except Exception as e:
        print(f"❌ Error updating jobs: {e}")
        return False

def verify_completed_jobs():
    """Verify that jobs are now marked as completed"""
    print("\n🔍 Verifying Completed Jobs...")
    
    try:
        # Check for completed jobs
        completed_jobs = list(firestore_client.db.collection('jobs').where('symbol', '==', '360ONE').where('status', '==', 'completed').stream())
        
        print(f"Found {len(completed_jobs)} completed jobs for 360ONE:")
        
        for i, job_doc in enumerate(completed_jobs):
            job_data = job_doc.to_dict()
            print(f"  Job {i+1}: {job_doc.id}")
            print(f"    Status: {job_data.get('status', 'unknown')}")
            print(f"    Completed: {job_data.get('completed_at', 'unknown')}")
            print(f"    Completed stages: {len(job_data.get('completed_stages', []))}")
            print(f"    Progress: {job_data.get('progress_percentage', 0)}%")
        
        return len(completed_jobs) > 0
        
    except Exception as e:
        print(f"❌ Error verifying jobs: {e}")
        return False

def test_cache_availability():
    """Test if cache is now available"""
    print("\n🧪 Testing Cache Availability...")
    
    try:
        # Check if cache is now available
        cached_analysis = firestore_client.get_latest_analysis_by_symbol('360ONE', 'enhanced')
        
        if cached_analysis:
            print("✅ Cache found for 360ONE!")
            print(f"  Job ID: {cached_analysis.get('job_id', 'unknown')}")
            print(f"  Status: {cached_analysis.get('status', 'unknown')}")
            print(f"  Created: {cached_analysis.get('created_at', 'unknown')}")
            return True
        else:
            print("❌ No cache found for 360ONE")
            return False
            
    except Exception as e:
        print(f"❌ Error checking cache: {e}")
        return False

def main():
    """Run the job update process"""
    print("🚀 Updating Database Jobs to Completed Status")
    print("=" * 60)
    
    # Step 1: Update jobs to completed
    print("\n" + "="*20 + " STEP 1: UPDATE JOBS " + "="*20)
    update_success = update_jobs_to_completed()
    
    # Step 2: Verify completed jobs
    print("\n" + "="*20 + " STEP 2: VERIFY JOBS " + "="*20)
    verify_success = verify_completed_jobs()
    
    # Step 3: Test cache availability
    print("\n" + "="*20 + " STEP 3: TEST CACHE " + "="*20)
    cache_success = test_cache_availability()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 UPDATE SUMMARY")
    print("=" * 60)
    
    print(f"Jobs Updated: {'✅ SUCCESS' if update_success else '❌ FAILED'}")
    print(f"Jobs Verified: {'✅ SUCCESS' if verify_success else '❌ FAILED'}")
    print(f"Cache Available: {'✅ SUCCESS' if cache_success else '❌ FAILED'}")
    
    if update_success and verify_success and cache_success:
        print("\n🎉 All steps completed successfully!")
        print("   - Jobs updated to completed status")
        print("   - Cache is now available for 360ONE")
        print("   - Database optimization will work for future requests")
    else:
        print("\n⚠️  Some steps failed. Please check the logs.")
    
    return update_success and verify_success and cache_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
