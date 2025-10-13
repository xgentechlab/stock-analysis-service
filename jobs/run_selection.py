#!/usr/bin/env python3
"""
Cron job wrapper for running daily stock selection
Can be called directly or via HTTP endpoint
"""
import sys
import os
import asyncio
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.selection import selection_engine
from app.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run the daily selection pipeline"""
    logger.info("Starting daily selection cron job")
    
    try:
        # Run selection
        summary = selection_engine.run_daily_selection()
        
        logger.info("Selection job completed successfully")
        logger.info(f"Summary: {summary}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Selection job failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
