#!/usr/bin/env python3
"""
Test runner for hot stocks optimization verification.
Runs all tests to verify the optimization changes work correctly.
"""

import sys
import os
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test(test_name: str, test_file: str) -> bool:
    """Run a single test and return success status"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {test_name}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            logger.info(f"✅ {test_name} PASSED")
            if result.stdout:
                logger.info("Output:")
                for line in result.stdout.strip().split('\n'):
                    logger.info(f"  {line}")
            return True
        else:
            logger.error(f"❌ {test_name} FAILED")
            if result.stderr:
                logger.error("Error output:")
                for line in result.stderr.strip().split('\n'):
                    logger.error(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {test_name} TIMED OUT")
        return False
    except Exception as e:
        logger.error(f"❌ {test_name} ERROR: {e}")
        return False

def main():
    """Run all optimization tests"""
    logger.info("Hot Stocks Optimization Test Suite")
    logger.info(f"Started at: {datetime.now()}")
    
    tests = [
        ("Quick Optimization Test", "quick_test_optimization.py"),
        ("Hot Stocks Endpoint Test", "test_hot_stocks_endpoint.py"),
        ("Comprehensive Optimization Test", "test_hot_stocks_optimization.py")
    ]
    
    results = {}
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            success = run_test(test_name, test_file)
            results[test_name] = success
        else:
            logger.warning(f"⚠️ Test file not found: {test_file}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUITE SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All optimization tests passed!")
        logger.info("✅ Hot stocks optimization is working correctly!")
        return True
    else:
        logger.error("💥 Some tests failed!")
        logger.error("❌ Please check the optimization changes!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
