import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def _format_enhanced_technical_indicators(technical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format technical indicators for storage (matches stage processor format)"""
    logger.info(f"🔍 FORMATTER: Processing technical data with keys: {list(technical_data.keys())}")
    logger.info(f"🔍 FORMATTER: Key indicators - sma20: {technical_data.get('sma20')}, rsi14: {technical_data.get('rsi14')}, vwap: {technical_data.get('vwap')}")
    
    return {
        "basic_indicators": {
            "sma_20": technical_data.get("sma20"),  # Fixed: was sma_20
            "sma_50": technical_data.get("sma50"),  # Fixed: was sma_50
            "rsi_14": technical_data.get("rsi14"),  # Fixed: was rsi_14
            "atr_14": technical_data.get("atr14"),  # Fixed: was atr_14
            "current_price": technical_data.get("current_price"),
            "close": technical_data.get("close"),
            # Support/Resistance indicators
            "high_52w": technical_data.get("high_52w"),
            "low_52w": technical_data.get("low_52w"),
            "recent_high": technical_data.get("recent_high"),
            "recent_low": technical_data.get("recent_low"),
            "vwap": technical_data.get("vwap")
        },
        "momentum_indicators": {
            "macd": technical_data.get("macd"),
            "macd_signal": technical_data.get("macd_signal"),
            "macd_histogram": technical_data.get("macd_histogram"),
            "stoch_rsi_k": technical_data.get("stoch_k"),  # Fixed: was stoch_rsi_k
            "stoch_rsi_d": technical_data.get("stoch_d"),  # Fixed: was stoch_rsi_d
            "williams_r": technical_data.get("williams_r"),
            "roc_5": technical_data.get("roc_5", None),  # Not calculated yet
            "roc_10": technical_data.get("roc_10", None),  # Not calculated yet
            "roc_20": technical_data.get("roc_20", None)  # Not calculated yet
        },
        "volume_indicators": {
            "vwap": technical_data.get("vwap", None),  # Not calculated yet
            "vwap_upper": technical_data.get("vwap_upper", None),  # Not calculated yet
            "vwap_lower": technical_data.get("vwap_lower", None),  # Not calculated yet
            "obv": technical_data.get("obv"),
            "ad_line": technical_data.get("ad_line", None)  # Not calculated yet
        },
        "divergence_signals": {
            "bullish_divergence": technical_data.get("bullish_divergence", False),
            "bearish_divergence": technical_data.get("bearish_divergence", False)
        },
        "multi_timeframe": {
            "1m_trend": technical_data.get("1m_trend", None),  # Not calculated yet
            "5m_trend": technical_data.get("5m_trend", None),  # Not calculated yet
            "15m_trend": technical_data.get("15m_trend", None),  # Not calculated yet
            "1d_trend": technical_data.get("1d_trend", None),  # Not calculated yet
            "1wk_trend": technical_data.get("1wk_trend", None)  # Not calculated yet
        }
    }


def format_optimized_analysis_response(analysis: dict) -> dict:
    """
    Format analysis response with specific optimizations:
    1. Eliminate duplicate top_drivers (keep only one instance)
    2. Hide internal calculations (calculation_steps, duration_seconds, blending_method)
    3. Better data grouping (recommendation, analysis_summary, risk_reward, key_metrics, drivers, technical_analysis)
    4. Simplify investment examples (keep only ₹10,000 example)
    5. Streamline metadata (minimal essential information only)
    
    Preserves: plain_english_summary, real_money_impacts, rationale explanations
    """
    try:
        if not analysis or not isinstance(analysis, dict):
            return analysis
            
        # Extract stage data
        stages = analysis.get("stages", {})
        simple_analysis = stages.get("simple_analysis", {}).get("data", {})
        simple_decision = stages.get("simple_decision", {}).get("data", {})
        verdict_synthesis = stages.get("verdict_synthesis", {}).get("data", {})
        final_scoring = stages.get("final_scoring", {}).get("data", {})
        
        # 1. RECOMMENDATION - All decision-related data in one place
        recommendation = {
            "action": verdict_synthesis.get("final_recommendation", {}).get("action", simple_decision.get("decision", "unknown")),
            "confidence": verdict_synthesis.get("final_recommendation", {}).get("confidence", simple_decision.get("confidence", 0.0)),
            "position_size": verdict_synthesis.get("final_recommendation", {}).get("position_size", simple_decision.get("position_size", "0%")),
            "target_price": simple_decision.get("target_price", 0.0),
            "stop_loss": verdict_synthesis.get("final_recommendation", {}).get("stop_loss", {}).get("price", simple_decision.get("stop_loss", 0.0)),
            "rationale": verdict_synthesis.get("final_recommendation", {}).get("rationale", simple_decision.get("reasoning", []))
        }
        
        # 2. ANALYSIS_SUMMARY - Key scores and signals
        analysis_summary = {
            "overall_signal": simple_analysis.get("overall_signal", "unknown"),
            "overall_signal_strength": simple_analysis.get("overall_signal_strength", 0.0),
            "fundamental_score": simple_analysis.get("fundamental_score", 0.0),
            "technical_score": simple_analysis.get("technical_score", 0.0),
            "final_score": final_scoring.get("final_score", 0.0),
            "risk_level": simple_decision.get("risk_level", "unknown"),
            "confidence": simple_analysis.get("confidence", 0.0)
        }
        
        # 3. RISK_REWARD - All risk/reward calculations grouped
        risk_reward_data = simple_analysis.get("risk_reward", {})
        risk_reward = {
            "ratio": simple_analysis.get("risk_reward_ratio", 0.0),
            "current_price": risk_reward_data.get("current_price", 0.0),
            "support_level": simple_analysis.get("support_level", 0.0),
            "resistance_level": risk_reward_data.get("resistance_level", 0.0),
            "downside_percentage": simple_analysis.get("downside_percentage", 0.0),
            "upside_percentage": simple_analysis.get("upside_percentage", 0.0),
            "downside": risk_reward_data.get("downside", 0.0),
            "upside": simple_analysis.get("upside", 0.0),
            "ratio_interpretation": risk_reward_data.get("ratio_interpretation", ""),
            # Include full real_money_impacts object
            "real_money_impacts": risk_reward_data.get("real_money_impacts", {}),
            # Keep only ₹10,000 example for quick reference
            "example_investment": risk_reward_data.get("real_money_impacts", {}).get("₹10,000", {}),
            # Preserve plain_english_summary
            "plain_english_summary": simple_analysis.get("plain_english_summary", {})
        }
        
        # 4. KEY_METRICS - Financial ratios in one section
        top_drivers = simple_analysis.get("top_drivers", {})
        drivers_list = top_drivers.get("drivers", [])
        
        # Extract key metrics from drivers
        key_metrics = {}
        for driver in drivers_list:
            metric = driver.get("metric", "")
            value = driver.get("value", "")
            if metric == "P/E Ratio":
                key_metrics["pe_ratio"] = value
            elif metric == "P/B Ratio":
                key_metrics["pb_ratio"] = value
            elif metric == "Operating Margin":
                key_metrics["operating_margin"] = value
            elif metric == "Current Ratio":
                key_metrics["current_ratio"] = value
            elif metric == "Debt/Equity":
                key_metrics["debt_equity"] = value
        
        # 5. DRIVERS - All factor analysis consolidated (eliminate duplicates)
        drivers = {
            "opportunity_factors": top_drivers.get("opportunity_factors", []),
            "risk_factors": top_drivers.get("risk_factors", []),
            "drivers": drivers_list,
            "decision_influence": top_drivers.get("decision_influence", "")
        }
        
        # 6. TECHNICAL_ANALYSIS - Technical indicators grouped
        technical_analysis = {
            "trend": simple_analysis.get("overall_signal", "unknown"),
            "support_levels": [simple_analysis.get("support_level", 0.0)],
            "resistance_levels": [risk_reward_data.get("resistance_level", 0.0)],
            "momentum": "weak" if simple_analysis.get("technical_score", 0.0) < 0.3 else "strong",
            "volume": "normal",  # Default since not explicitly provided
            "the_setup": simple_analysis.get("the_setup", ""),
            "the_catalyst": simple_analysis.get("the_catalyst", ""),
            "the_confirmation": simple_analysis.get("the_confirmation", ""),
            "primary_driver": simple_analysis.get("primary_driver", "")
        }
        
        # 7. METADATA - Minimal essential information only
        metadata = {
            "analysis_date": analysis.get("completed_at", ""),
            "freshness": "fresh",  # Simplified
            "stages_completed": len(analysis.get("completed_stages", [])),
            "analysis_type": analysis.get("analysis_type", "enhanced")
        }
        
        # Build optimized response
        optimized_response = {
            "symbol": analysis.get("symbol", ""),
            "analysis_type": analysis.get("analysis_type", "enhanced"),
            "recommendation": recommendation,
            "analysis_summary": analysis_summary,
            "risk_reward": risk_reward,
            "key_metrics": key_metrics,
            "drivers": drivers,
            "technical_analysis": technical_analysis,
            "metadata": metadata
        }
        
        logger.info(f"🔧 Optimized analysis response formatted for {analysis.get('symbol', 'unknown')}")
        return optimized_response
        
    except Exception as e:
        logger.error(f"Error formatting optimized analysis response: {e}")
        return analysis
