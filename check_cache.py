#!/usr/bin/env python3
"""
Check if there's cached analysis for a symbol
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def check_cached_analysis(symbol):
    """Check if there's cached analysis for a symbol"""
    
    print(f"🔍 Checking cached analysis for {symbol}")
    print("=" * 50)
    
    try:
        # Check direct analysis endpoint
        response = requests.get(f"{BASE_URL}/analysis/{symbol}")
        
        if response.status_code == 200:
            data = response.json()
            if data["ok"]:
                analysis = data["data"]["analysis"]
                freshness = data["data"]["freshness"]
                
                print(f"✅ Found cached analysis for {symbol}")
                print(f"   Created: {analysis['created_at']}")
                print(f"   Status: {analysis['status']}")
                print(f"   Fresh: {freshness['is_fresh']}")
                print(f"   Age: {freshness['age_days']} days")
                print(f"   Freshness Score: {freshness['freshness_score']}")
                print(f"   Recommendation: {data['data']['recommendation']}")
                
                if freshness['stale_stages']:
                    print(f"   Stale Stages: {[s['stage'] for s in freshness['stale_stages']]}")
                else:
                    print(f"   All stages are fresh!")
                    
            else:
                print(f"❌ No cached analysis found: {data['error']}")
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def check_analysis_history(symbol):
    """Check analysis history for a symbol"""
    
    print(f"\n📈 Checking analysis history for {symbol}")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/analysis/{symbol}/history?days_back=30&limit=5")
        
        if response.status_code == 200:
            data = response.json()
            if data["ok"]:
                analyses = data["data"]["analyses"]
                print(f"Found {len(analyses)} analyses in last 30 days")
                
                for i, analysis in enumerate(analyses):
                    freshness = analysis["freshness"]
                    print(f"   {i+1}. {analysis['created_at']} - Fresh: {freshness['is_fresh']} ({freshness['age_days']} days)")
            else:
                print(f"❌ No history found: {data['error']}")
        else:
            print(f"❌ History API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ History request failed: {e}")

if __name__ == "__main__":
    print("🚀 Cache Analysis Checker")
    print("=" * 60)
    
    # Check for RELIANCE
    check_cached_analysis("RELIANCE")
    check_analysis_history("RELIANCE")
    
    print(f"\n✅ Check completed!")
