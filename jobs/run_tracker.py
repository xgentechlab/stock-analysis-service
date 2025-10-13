#!/usr/bin/env python3
"""
Cron job wrapper for running position tracker
Can be called directly or via HTTP endpoint
"""
import sys
import os
import asyncio
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tracker import position_tracker
from app.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run the position tracking cycle"""
    logger.info("Starting position tracker cron job")
    
    try:
        # Run tracker
        summary = position_tracker.run_position_tracking()
        
        logger.info("Tracker job completed successfully")
        logger.info(f"Summary: {summary}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Tracker job failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
