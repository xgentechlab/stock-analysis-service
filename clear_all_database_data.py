#!/usr/bin/env python3
"""
Comprehensive script to clear ALL data from Firestore database
This script will remove all documents from ALL collections in the database.

WARNING: This will permanently delete ALL data from the database!
Use with extreme caution - this action cannot be undone.

Collections that will be cleared:
- jobs (analysis jobs)
- signals (trading signals)
- positions (trading positions)
- fills (trade fills)
- audits (audit logs)
- configs (runtime configurations)
- recommendations (stock recommendations)
- watchlist (user watchlists)
- portfolio (user portfolios)
- user_decisions (user trading decisions)
- portfolio_suggestions (portfolio suggestions)
- multi_timeframe_analyses (multi-timeframe technical analysis data)
- hot_stock_analyses (hot stock analysis data)
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

class CompleteDatabaseCleaner:
    """Utility class to clean ALL data from Firestore database"""
    
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
            "portfolio_suggestions",
            "multi_timeframe_analyses",
            "hot_stock_analyses"
        ]
        
        # Collections that are safe to delete (user data)
        self.user_data_collections = [
            "recommendations",
            "watchlist",
            "portfolio",
            "user_decisions",
            "portfolio_suggestions"
        ]
        
        # Collections that are system data
        self.system_data_collections = [
            "jobs",
            "signals",
            "positions",
            "fills",
            "audits",
            "configs",
            "multi_timeframe_analyses",
            "hot_stock_analyses"
        ]
        
    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection"""
        try:
            docs = list(self.db.collection(collection_name).stream())
            return len(docs)
        except Exception as e:
            logger.error(f"Failed to count documents in {collection_name}: {e}")
            return 0
    
    def delete_collection_batch(self, collection_name: str, batch_size: int = 50) -> int:
        """Delete all documents in a collection using smaller batch operations to avoid transaction limits"""
        try:
            deleted_count = 0
            batch_count = 0
            
            while True:
                # Get a smaller batch of documents to avoid transaction size limits
                docs = list(self.db.collection(collection_name).limit(batch_size).stream())
                
                if not docs:
                    break
                
                # Delete documents in smaller batches
                batch = self.db.batch()
                batch_operations = 0
                
                for doc in docs:
                    try:
                        batch.delete(doc.reference)
                        batch_operations += 1
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"  Failed to add document {doc.id} to batch: {e}")
                        continue
                
                # Commit the batch with error handling
                try:
                    if batch_operations > 0:
                        batch.commit()
                        batch_count += 1
                        logger.info(f"  Deleted batch {batch_count} of {batch_operations} documents from {collection_name}")
                    else:
                        logger.warning(f"  No valid documents to delete in batch for {collection_name}")
                        break
                except Exception as e:
                    if "Transaction too big" in str(e):
                        # Reduce batch size and retry
                        logger.warning(f"  Transaction too big, reducing batch size from {batch_size} to {batch_size//2}")
                        batch_size = max(10, batch_size // 2)
                        continue
                    else:
                        logger.error(f"  Failed to commit batch for {collection_name}: {e}")
                        # Try individual deletions as fallback
                        self._delete_documents_individually(docs)
                        deleted_count += len(docs)
                        break
                
                # Add small delay to avoid overwhelming Firestore
                import time
                time.sleep(0.1)
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return 0
    
    def _delete_documents_individually(self, docs) -> int:
        """Fallback method to delete documents one by one when batch operations fail"""
        deleted_count = 0
        for doc in docs:
            try:
                doc.reference.delete()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"  Failed to delete individual document {doc.id}: {e}")
        return deleted_count
    
    def get_database_summary(self) -> Dict[str, int]:
        """Get summary of all collections and their document counts"""
        summary = {}
        
        logger.info("📊 Analyzing ALL database collections...")
        
        for collection_name in self.all_collections:
            count = self.get_collection_count(collection_name)
            summary[collection_name] = count
            
            # Categorize collections
            if collection_name in self.user_data_collections:
                category = "👤 User Data"
            elif collection_name in self.system_data_collections:
                category = "⚙️  System Data"
            else:
                category = "📁 Other"
            
            logger.info(f"  {category} - {collection_name}: {count} documents")
        
        return summary
    
    def confirm_deletion(self, summary: Dict[str, int]) -> bool:
        """Ask user to confirm deletion with detailed breakdown"""
        total_docs = sum(summary.values())
        
        if total_docs == 0:
            logger.info("✅ No documents found to delete.")
            return False
        
        print(f"\n{'='*60}")
        print(f"⚠️  CRITICAL WARNING: COMPLETE DATABASE WIPE")
        print(f"{'='*60}")
        print(f"Total documents to be deleted: {total_docs}")
        print(f"\nCollections to be cleared:")
        
        # Show user data collections
        user_docs = 0
        print(f"\n👤 USER DATA COLLECTIONS:")
        for collection in self.user_data_collections:
            count = summary.get(collection, 0)
            if count > 0:
                print(f"  - {collection}: {count} documents")
                user_docs += count
        if user_docs == 0:
            print("  - No user data found")
        
        # Show system data collections
        system_docs = 0
        print(f"\n⚙️  SYSTEM DATA COLLECTIONS:")
        for collection in self.system_data_collections:
            count = summary.get(collection, 0)
            if count > 0:
                print(f"  - {collection}: {count} documents")
                system_docs += count
        if system_docs == 0:
            print("  - No system data found")
        
        print(f"\n📊 SUMMARY:")
        print(f"  - User data: {user_docs} documents")
        print(f"  - System data: {system_docs} documents")
        print(f"  - Total: {total_docs} documents")
        
        print(f"\n{'='*60}")
        print(f"🚨 THIS ACTION CANNOT BE UNDONE! 🚨")
        print(f"{'='*60}")
        
        # Double confirmation
        print(f"\nTo proceed, you must type 'DELETE ALL DATA' exactly:")
        response1 = input("Confirmation 1: ").strip()
        
        if response1 != "DELETE ALL DATA":
            print("❌ First confirmation failed. Operation cancelled.")
            return False
        
        print(f"\nAre you absolutely sure? Type 'YES' to proceed:")
        response2 = input("Confirmation 2: ").strip()
        
        if response2 != "YES":
            print("❌ Second confirmation failed. Operation cancelled.")
            return False
        
        return True
    
    def clean_database(self, collections: List[str] = None) -> Dict[str, int]:
        """Clean specified collections from the database with robust error handling"""
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
            
            # For very large collections, use recursive deletion
            if initial_count > 1000:
                logger.info(f"  Large collection detected ({initial_count} docs), using recursive deletion...")
                deleted_count = self._delete_collection_recursive(collection_name)
            else:
                # Use batch deletion for smaller collections
                deleted_count = self.delete_collection_batch(collection_name)
            
            results[collection_name] = deleted_count
            
            # Verify deletion
            final_count = self.get_collection_count(collection_name)
            
            if final_count == 0:
                logger.info(f"  ✅ {collection_name}: Successfully deleted {deleted_count} documents")
            else:
                logger.warning(f"  ⚠️  {collection_name}: Deleted {deleted_count} documents, {final_count} remaining")
                
                # If there are still documents, try individual deletion as last resort
                if final_count > 0 and final_count < 100:
                    logger.info(f"  Attempting individual deletion for remaining {final_count} documents...")
                    remaining_deleted = self._delete_remaining_documents(collection_name)
                    results[collection_name] += remaining_deleted
                    logger.info(f"  Additional {remaining_deleted} documents deleted individually")
        
        return results
    
    def _delete_collection_recursive(self, collection_name: str) -> int:
        """Recursively delete all documents in a collection using very small batches"""
        deleted_count = 0
        batch_size = 10  # Very small batches for large collections
        
        while True:
            docs = list(self.db.collection(collection_name).limit(batch_size).stream())
            
            if not docs:
                break
            
            # Delete in very small batches
            batch = self.db.batch()
            for doc in docs:
                batch.delete(doc.reference)
            
            try:
                batch.commit()
                deleted_count += len(docs)
                logger.info(f"  Recursive batch: Deleted {len(docs)} documents (total: {deleted_count})")
            except Exception as e:
                logger.warning(f"  Recursive batch failed: {e}")
                # Fall back to individual deletion
                for doc in docs:
                    try:
                        doc.reference.delete()
                        deleted_count += 1
                    except Exception as e2:
                        logger.warning(f"  Individual delete failed for {doc.id}: {e2}")
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(0.2)
        
        return deleted_count
    
    def _delete_remaining_documents(self, collection_name: str) -> int:
        """Delete any remaining documents individually"""
        deleted_count = 0
        docs = list(self.db.collection(collection_name).stream())
        
        for doc in docs:
            try:
                doc.reference.delete()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"  Failed to delete remaining document {doc.id}: {e}")
        
        return deleted_count
    
    def create_audit_log(self, action: str, details: Dict[str, Any]) -> Optional[str]:
        """Create an audit log for the cleanup operation"""
        try:
            audit_id = firestore_client.create_audit_log(
                action=action,
                details=details,
                source="complete_cleanup_script"
            )
            logger.info(f"📝 Created audit log: {audit_id}")
            return audit_id
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            return None

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
        
        logger.info("🚀 Complete Database Cleanup Script Started")
        logger.info(f"📋 Target project: {settings.firestore_project_id}")
        logger.info(f"📋 Target database: {settings.firestore_database_id}")
        
        # Initialize cleaner
        cleaner = CompleteDatabaseCleaner()
        
        # Get database summary
        summary = cleaner.get_database_summary()
        
        # Check if there's anything to delete
        total_docs = sum(summary.values())
        if total_docs == 0:
            logger.info("✅ No documents found to delete. Database is already clean.")
            return 0
        
        # Confirm deletion
        if not cleaner.confirm_deletion(summary):
            logger.info("❌ Cleanup cancelled by user")
            return 0
        
        # Perform cleanup
        logger.info("\n🧹 Starting deletion process...")
        results = cleaner.clean_database()
        
        # Create audit log
        audit_details = {
            "collections_cleaned": results,
            "total_documents_deleted": sum(results.values()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script_version": "2.0.0",
            "cleanup_type": "complete_database_wipe"
        }
        
        cleaner.create_audit_log("complete_database_cleanup", audit_details)
        
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
        
        # Note about storage optimization
        if 'multi_timeframe_analyses' in results and results['multi_timeframe_analyses'] > 0:
            logger.info(f"💾 Note: multi_timeframe_analyses documents are now optimized (no timeframes data)")
            logger.info(f"   This significantly reduces storage usage compared to previous versions.")
        
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
