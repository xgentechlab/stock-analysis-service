#!/usr/bin/env python3
"""
Script to check if multi-timeframe analysis data is being stored in the database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.firestore_client import firestore_client
import json
from datetime import datetime

def check_analysis_data():
    """Check what analysis data is stored in the database"""
    try:
        print("🔍 Checking multi-timeframe analysis data in database...")
        
        # Check multi-timeframe analyses
        print("\n📊 Multi-timeframe Analyses:")
        mtf_analyses = firestore_client.list_multi_timeframe_analyses(limit=10)
        print(f"Found {len(mtf_analyses)} multi-timeframe analyses")
        
        for i, analysis in enumerate(mtf_analyses[:3], 1):
            symbol = analysis.get('symbol', 'Unknown')
            analysis_id = analysis.get('analysis_id', 'Unknown')
            created_at = analysis.get('created_at', 'Unknown')
            timeframes = analysis.get('timeframes', {})
            mtf_score = analysis.get('mtf_score', 'N/A')
            
            print(f"\n{i}. {symbol} (ID: {analysis_id[:8]}...)")
            print(f"   Created: {created_at}")
            print(f"   MTF Score: {mtf_score}")
            print(f"   Timeframes: {list(timeframes.keys())}")
            
            # Show sample data for first timeframe
            if timeframes:
                first_tf = list(timeframes.keys())[0]
                tf_data = timeframes[first_tf]
                data_points = tf_data.get('data_points', 0)
                print(f"   {first_tf} data points: {data_points}")
        
        # Check hot stock analyses
        print("\n🔥 Hot Stock Analyses:")
        hot_analyses = firestore_client.list_hot_stock_analyses(limit=10)
        print(f"Found {len(hot_analyses)} hot stock analyses")
        
        for i, analysis in enumerate(hot_analyses[:3], 1):
            symbol = analysis.get('symbol', 'Unknown')
            analysis_id = analysis.get('analysis_id', 'Unknown')
            created_at = analysis.get('created_at', 'Unknown')
            scores = analysis.get('scores', {})
            technical = analysis.get('technical_indicators', {})
            enhanced_fundamentals = analysis.get('enhanced_fundamentals', {})
            
            print(f"\n{i}. {symbol} (ID: {analysis_id[:8]}...)")
            print(f"   Created: {created_at}")
            print(f"   Basic Composite Score: {scores.get('composite_score', 'N/A')}")
            print(f"   Basic Fundamental Score: {scores.get('fundamental_score', 'N/A')}")
            print(f"   Enhanced Technical Score: {scores.get('enhanced_technical_score', 'N/A')}")
            print(f"   Enhanced Fundamental Score: {scores.get('enhanced_fundamental_score', 'N/A')}")
            print(f"   Enhanced Combined Score: {scores.get('enhanced_combined_score', 'N/A')}")
            print(f"   Technical Confidence: {scores.get('enhanced_technical_confidence', 'N/A')}")
            print(f"   Technical Strength: {scores.get('enhanced_technical_strength', 'N/A')}")
            print(f"   Momentum: {scores.get('momentum_pct_5d', 'N/A')}%")
            print(f"   Volume Spike: {scores.get('volume_spike_ratio', 'N/A')}x")
            print(f"   RSI: {technical.get('rsi', 'N/A')} (Enhanced: {technical.get('enhanced_rsi', 'N/A')})")
            print(f"   MACD: {technical.get('macd', 'N/A')} (Enhanced: {technical.get('enhanced_macd', 'N/A')})")
            print(f"   Stage 1&2 Integrated: {analysis.get('stage_1_2_integrated', False)}")
            print(f"   Data Fetch Optimized: {analysis.get('data_fetch_optimized', False)}")
            print(f"   MTF Analysis ID: {analysis.get('multi_timeframe_analysis_id', 'N/A')[:8]}...")
            if enhanced_fundamentals:
                quality_metrics = enhanced_fundamentals.get('quality_metrics', {})
                print(f"   ROE Consistency: {quality_metrics.get('roe_consistency', 'N/A')}")
                print(f"   Debt/Equity: {quality_metrics.get('debt_equity_ratio', 'N/A')}")
                print(f"   Interest Coverage: {quality_metrics.get('interest_coverage', 'N/A')}")

        # Check jobs collection
        print("\n🔄 Jobs Collection:")
        jobs = firestore_client.list_jobs(limit=10)
        print(f"Found {len(jobs)} jobs")
        
        for i, job in enumerate(jobs[:3], 1):
            symbol = job.get('symbol', 'Unknown')
            status = job.get('status', 'Unknown')
            analysis_type = job.get('analysis_type', 'Unknown')
            created_at = job.get('created_at', 'Unknown')
            
            print(f"\n{i}. {symbol} - {analysis_type} ({status})")
            print(f"   Created: {created_at}")
            
            # Check if job has stages
            stages = job.get('stages', {})
            if stages:
                print(f"   Stages: {list(stages.keys())}")
                
                # Check data collection stage
                data_stage = stages.get('data_collection_and_analysis')
                if data_stage and data_stage.get('data'):
                    stage_data = data_stage['data']
                    has_enhanced_technical = 'enhanced_technical' in stage_data
                    has_enhanced_fundamentals = 'enhanced_fundamentals' in stage_data
                    has_mtf_id = 'multi_timeframe_analysis_id' in stage_data
                    
                    print(f"   Data Collection Stage:")
                    print(f"     - Enhanced Technical: {has_enhanced_technical}")
                    print(f"     - Enhanced Fundamentals: {has_enhanced_fundamentals}")
                    print(f"     - MTF Analysis ID: {has_mtf_id}")
                    if has_mtf_id:
                        mtf_id = stage_data.get('multi_timeframe_analysis_id', '')
                        print(f"     - MTF ID: {mtf_id[:8]}...")
        
        print("\n✅ Database check completed!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_analysis_data()
