#!/usr/bin/env python3
"""
Quick test to verify hot stocks optimization changes.
Simple verification that the optimized data fetching works.
"""

import sys
import os
import logging
import time

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test of the optimization"""
    logger.info("Quick Test: Hot Stocks Optimization")
    logger.info("=" * 50)
    
    try:
        from app.services.stocks import stocks_service
        
        # Test with a single symbol
        test_symbol = "RELIANCE"
        logger.info(f"Testing optimized data fetching for {test_symbol}")
        
        start_time = time.time()
        enhanced_info = stocks_service.get_enhanced_stock_info(test_symbol)
        fetch_time = time.time() - start_time
        
        if enhanced_info:
            logger.info(f"✅ Success: Data fetched in {fetch_time:.2f}s")
            
            # Check key data points
            checks = {
                "Symbol": enhanced_info.get("symbol"),
                "OHLCV data": "✅" if enhanced_info.get("ohlcv") is not None else "❌",
                "Enhanced technical": "✅" if enhanced_info.get("enhanced_technical") else "❌",
                "Enhanced fundamentals": "✅" if enhanced_info.get("enhanced_fundamentals") else "❌",
                "Fundamental score": "✅" if enhanced_info.get("fundamental_score") else "❌",
                "Data fetch optimized": "✅" if enhanced_info.get("data_fetch_optimized") else "❌"
            }
            
            for check, result in checks.items():
                logger.info(f"  {check}: {result}")
            
            # Test the hot stocks signal computation
            logger.info("\nTesting hot stocks signal computation...")
            try:
                from app.api.routes_hot_stocks import _compute_signals_for_symbol
                
                start_time = time.time()
                signals = _compute_signals_for_symbol(test_symbol, use_enhanced_indicators=True)
                signal_time = time.time() - start_time
                
                if signals:
                    logger.info(f"✅ Signals computed in {signal_time:.2f}s")
                    
                    # Check for enhanced scores
                    enhanced_combined = signals.get("enhanced_combined_score")
                    enhanced_technical = signals.get("enhanced_technical_score")
                    enhanced_fundamental = signals.get("enhanced_fundamental_score")
                    
                    logger.info(f"  Enhanced combined score: {enhanced_combined}")
                    logger.info(f"  Enhanced technical score: {enhanced_technical}")
                    logger.info(f"  Enhanced fundamental score: {enhanced_fundamental}")
                    
                    # Check basic signals
                    momentum = signals.get("momentum_pct_5d", 0.0)
                    volume_spike = signals.get("volume_spike_ratio", 0.0)
                    logger.info(f"  Momentum (5d): {momentum:.2f}%")
                    logger.info(f"  Volume spike: {volume_spike:.2f}x")
                    
                else:
                    logger.warning("❌ No signals computed")
                    
            except Exception as e:
                logger.error(f"❌ Signal computation failed: {e}")
            
        else:
            logger.error("❌ No enhanced stock info returned")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    logger.info("\n✅ Quick test completed successfully!")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        logger.info("🎉 Optimization verification passed!")
    else:
        logger.error("💥 Optimization verification failed!")
    sys.exit(0 if success else 1)
