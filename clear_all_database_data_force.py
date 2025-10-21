#!/usr/bin/env python3
"""
FORCE script to clear ALL data from Firestore database
This script will remove all documents from ALL collections in the database.

WARNING: This will permanently delete ALL data from the database!
Use with extreme caution - this action cannot be undone.

This version runs without confirmation prompts for automated use.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
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

class ForceDatabaseCleaner:
    """Utility class to clean ALL data from Firestore database without confirmation"""
    
    def __init__(self):
        self.db = firestore_client.db
        # All collections in the database
        self.all_collections = [
            "jobs",
            "signals", 
            "positions",
            "fills",
            "audits",
            "configs",
            "recommendations",
            "watchlist",
            "portfolio",
            "user_decisions",
            "portfolio_suggestions"
        ]
        
    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection"""
        try:
            docs = list(self.db.collection(collection_name).stream())
            return len(docs)
        except Exception as e:
            logger.error(f"Failed to count documents in {collection_name}: {e}")
            return 0
    
    def delete_collection_batch(self, collection_name: str, batch_size: int = 100) -> int:
        """Delete all documents in a collection using batch operations"""
        try:
            deleted_count = 0
            
            while True:
                # Get a batch of documents
                docs = list(self.db.collection(collection_name).limit(batch_size).stream())
                
                if not docs:
                    break
                
                # Delete documents in batch
                batch = self.db.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                    deleted_count += 1
                
                # Commit the batch
                batch.commit()
                logger.info(f"  Deleted batch of {len(docs)} documents from {collection_name}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return 0
    
    def get_database_summary(self) -> Dict[str, int]:
        """Get summary of all collections and their document counts"""
        summary = {}
        
        logger.info("📊 Analyzing ALL database collections...")
        
        for collection_name in self.all_collections:
            count = self.get_collection_count(collection_name)
            summary[collection_name] = count
            logger.info(f"  {collection_name}: {count} documents")
        
        return summary
    
    def clean_database(self, collections: List[str] = None) -> Dict[str, int]:
        """Clean specified collections from the database"""
        if collections is None:
            collections = self.all_collections
        
        results = {}
        
        logger.info("🧹 Starting complete database cleanup...")
        
        for collection_name in collections:
            logger.info(f"🗑️  Cleaning collection: {collection_name}")
            
            # Get count before deletion
            initial_count = self.get_collection_count(collection_name)
            
            if initial_count == 0:
                logger.info(f"  {collection_name}: No documents to delete")
                results[collection_name] = 0
                continue
            
            # Delete all documents
            deleted_count = self.delete_collection_batch(collection_name)
            results[collection_name] = deleted_count
            
            # Verify deletion
            final_count = self.get_collection_count(collection_name)
            
            if final_count == 0:
                logger.info(f"  ✅ {collection_name}: Successfully deleted {deleted_count} documents")
            else:
                logger.warning(f"  ⚠️  {collection_name}: Deleted {deleted_count} documents, {final_count} remaining")
        
        return results

def main():
    """Main function to run the complete database cleanup"""
    try:
        # Check if Firestore is properly configured
        if not settings.firestore_project_id:
            logger.error("❌ Firestore project ID not configured!")
            logger.error("Please set FIRESTORE_PROJECT_ID environment variable")
            return 1
        
        if not firestore_client.db:
            logger.error("❌ Firestore client not initialized!")
            logger.error("Please check your Google Cloud credentials")
            return 1
        
        logger.info("🚀 FORCE Database Cleanup Script Started")
        logger.info(f"📋 Target project: {settings.firestore_project_id}")
        logger.info(f"📋 Target database: {settings.firestore_database_id}")
        logger.warning("⚠️  RUNNING IN FORCE MODE - NO CONFIRMATION REQUIRED")
        
        # Initialize cleaner
        cleaner = ForceDatabaseCleaner()
        
        # Get database summary
        summary = cleaner.get_database_summary()
        
        # Check if there's anything to delete
        total_docs = sum(summary.values())
        if total_docs == 0:
            logger.info("✅ No documents found to delete. Database is already clean.")
            return 0
        
        logger.info(f"📊 Found {total_docs} documents to delete across {len([c for c, count in summary.items() if count > 0])} collections")
        
        # Perform cleanup
        logger.info("\n🧹 Starting deletion process...")
        results = cleaner.clean_database()
        
        # Summary
        logger.info("\n📊 Cleanup Summary:")
        total_deleted = 0
        for collection, count in results.items():
            if count > 0:
                logger.info(f"  ✅ {collection}: {count} documents deleted")
                total_deleted += count
            else:
                logger.info(f"  ⏭️  {collection}: No documents found")
        
        logger.info(f"\n🎉 Complete database cleanup completed successfully!")
        logger.info(f"📈 Total documents deleted: {total_deleted}")
        logger.info(f"🗃️  Collections cleared: {len([c for c, count in results.items() if count > 0])}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n❌ Cleanup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
