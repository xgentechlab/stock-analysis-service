"""
Multi-Stage Prompting System for Enhanced Stock Analysis
Implements a 4-stage approach to prevent AI from defaulting to "watch" and provide more decisive analysis
"""
import json
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from app.services.openai_client import openai_client
from app.services.claude_client import claude_client
from app.config import settings

logger = logging.getLogger(__name__)

class AnalysisModule(Enum):
    """Specialized analysis modules based on score patterns"""
    MOMENTUM = "momentum"
    VALUE_ENTRY = "value_entry"
    BALANCED = "balanced"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class MultiStagePromptingService:
    """
    Multi-stage prompting system that breaks down analysis into specialized stages
    to prevent AI from defaulting to "watch" and provide more decisive recommendations
    """
    
    def __init__(self):
        self.openai_client = openai_client
        self.claude_client = claude_client
        # AI Provider selection - can be "openai" or "claude"
        self.ai_provider = getattr(settings, 'ai_provider', 'openai')
        
    def analyze_stock(self, symbol: str, fundamentals: Dict[str, Any], 
                     technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for multi-stage analysis
        Returns comprehensive analysis with all stages
        """
        try:
            logger.info(f"🔍 Multi-stage analysis for {symbol} using {self.ai_provider.upper()}")
            
            # Stage 1: Forensic Analysis
            forensic_analysis = self._stage1_forensic_analysis(symbol, fundamentals, technical, enhanced_fundamentals)
            if not forensic_analysis:
                return {"error": "Stage 1 analysis failed"}
            
            # Stage 2: Module Selection and Specialized Analysis
            module_analysis = self._stage2_module_selection(symbol, forensic_analysis, fundamentals, technical, enhanced_fundamentals)
            if not module_analysis:
                return {"error": "Stage 2 analysis failed"}
            
            # Stage 3: Risk Assessment
            risk_assessment = self._stage3_risk_assessment(symbol, forensic_analysis, module_analysis, fundamentals, technical)
            if not risk_assessment:
                return {"error": "Stage 3 analysis failed"}
            
            # Stage 4: Final Decision
            final_decision = self._stage4_final_decision(symbol, forensic_analysis, module_analysis, risk_assessment, fundamentals, technical)
            if not final_decision:
                return {"error": "Stage 4 analysis failed"}
            
            # Compile comprehensive result
            result = {
                "symbol": symbol,
                "analysis_stages": {
                    "stage1_forensic": forensic_analysis,
                    "stage2_module": module_analysis,
                    "stage3_risk": risk_assessment,
                    "stage4_decision": final_decision
                },
                "final_recommendation": {
                    "action": final_decision["action"],
                    "confidence": final_decision["confidence"],
                    "rationale": final_decision["rationale"],
                    "position_size": final_decision.get("position_size", "50%"),
                    "stop_loss": final_decision.get("stop_loss"),
                    "target_price": final_decision.get("target_price"),
                    "holding_period": final_decision.get("holding_period", "medium-term")
                },
                "risk_summary": {
                    "risk_level": risk_assessment["risk_level"],
                    "deal_breakers": risk_assessment.get("deal_breakers", []),
                    "manageable_risks": risk_assessment.get("manageable_risks", [])
                }
            }
            
            logger.info(f"✅ Multi-stage analysis completed for {symbol}: {final_decision['action']} (confidence: {final_decision['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-stage analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def _stage1_forensic_analysis(self, symbol: str, fundamentals: Dict[str, Any], 
                                 technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Forensic Analysis Prompt
        Identifies top bullish/bearish factors, support/resistance, momentum direction, etc.
        """
        try:
            # Anonymize data
            anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
            
            prompt = f"""
You are a forensic financial analyst. Analyze this comprehensive stock data and identify the most critical factors.

=== STOCK DATA ===
{self._format_data_for_analysis(anonymized_data)}

=== FORENSIC ANALYSIS TASK ===

Analyze ALL the data and identify:

1. TOP 3 BULLISH FACTORS (with specific numbers):
   - Factor 1: [Specific metric and value]
   - Factor 2: [Specific metric and value] 
   - Factor 3: [Specific metric and value]

2. TOP 3 BEARISH FACTORS (with specific numbers):
   - Factor 1: [Specific metric and value]
   - Factor 2: [Specific metric and value]
   - Factor 3: [Specific metric and value]

3. KEY SUPPORT/RESISTANCE LEVELS:
   - Support Level 1: [Price and reasoning]
   - Support Level 2: [Price and reasoning]
   - Resistance Level 1: [Price and reasoning]
   - Resistance Level 2: [Price and reasoning]

4. TIMEFRAME ALIGNMENT STRENGTH (0-1):
   - Score: [0.0-1.0]
   - Reasoning: [Which timeframes align and how strong]

5. MOMENTUM DIRECTION:
   - Direction: [accelerating/steady/decelerating]
   - Strength: [weak/moderate/strong]
   - Confirmation: [volume/technical/fundamental]

6. VOLUME CONFIRMATION QUALITY (0-1):
   - Score: [0.0-1.0]
   - Analysis: [Volume patterns and institutional interest]

7. FUNDAMENTAL vs TECHNICAL SCORE:
   - Fundamental Score: [0.0-1.0]
   - Technical Score: [0.0-1.0]
   - Dominant Factor: [fundamental/technical/balanced]

8. RISK-REWARD CALCULATION:
   - Current Price: [price]
   - Resistance Level: [price and source]
   - Support Level: [price and source]
   - Upside: [amount and percentage]
   - Downside: [amount and percentage]
   - Risk-Reward Ratio: [X:1 format]
   - Interpretation: [excellent/good/acceptable/poor]

9. OVERALL SIGNAL STRENGTH (0-1):
   - Score: [0.0-1.0]
   - Primary Driver: [momentum/value/breakout/quality]

CRITICAL: For risk-reward calculation, use this formula:
- Upside = (Resistance - Current Price)
- Downside = (Current Price - Support)
- Ratio = Upside / Downside
- Express as X:1 (e.g., "2.5:1" means risk ₹1 to make ₹2.5)
- NEVER express as inverse (22:1 is WRONG if you mean 1:22)

Return as JSON:
{{
    "bullish_factors": ["factor1", "factor2", "factor3"],
    "bearish_factors": ["factor1", "factor2", "factor3"],
    "support_levels": [{{"price": 0, "reasoning": "text"}}],
    "resistance_levels": [{{"price": 0, "reasoning": "text"}}],
    "timeframe_alignment": {{"score": 0.0, "reasoning": "text"}},
    "momentum_direction": {{"direction": "accelerating", "strength": "strong", "confirmation": "volume"}},
    "volume_confirmation": {{"score": 0.0, "analysis": "text"}},
    "fundamental_score": 0.0,
    "technical_score": 0.0,
    "dominant_factor": "technical",
    "risk_reward_calculation": {{
        "current_price": 0.0,
        "resistance_level": {{"price": 0.0, "source": "text"}},
        "support_level": {{"price": 0.0, "source": "text"}},
        "upside": {{"amount": 0.0, "percentage": 0.0}},
        "downside": {{"amount": 0.0, "percentage": 0.0}},
        "risk_reward_ratio": 0.0,
        "interpretation": "text"
    }},
    "overall_signal_strength": 0.0,
    "primary_driver": "momentum"
}}
"""
            
            response = self._call_ai_provider(prompt, "You are a forensic financial analyst specializing in detailed stock analysis.")
            if not response:
                return None
                
            return self._parse_json_response(response, "Stage 1 Forensic Analysis")
            
        except Exception as e:
            logger.error(f"Stage 1 analysis failed for {symbol}: {e}")
            return None
    
    def _stage2_module_selection(self, symbol: str, forensic_analysis: Dict[str, Any], 
                                fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                                enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Specialized Module Selection
        Routes to Momentum, Value Entry, or Balanced module based on scores
        """
        try:
            technical_score = forensic_analysis.get("technical_score", 0.0)
            fundamental_score = forensic_analysis.get("fundamental_score", 0.0)
            
            # Determine which module to use
            if technical_score > 0.7:
                module = AnalysisModule.MOMENTUM
                analysis = self._momentum_module_analysis(symbol, forensic_analysis, fundamentals, technical, enhanced_fundamentals)
            elif fundamental_score > 0.7 and technical_score < 0.6:
                module = AnalysisModule.VALUE_ENTRY
                analysis = self._value_entry_module_analysis(symbol, forensic_analysis, fundamentals, technical, enhanced_fundamentals)
            else:
                module = AnalysisModule.BALANCED
                analysis = self._balanced_module_analysis(symbol, forensic_analysis, fundamentals, technical, enhanced_fundamentals)
            
            if not analysis:
                return None
                
            return {
                "selected_module": module.value,
                "module_analysis": analysis,
                "selection_reasoning": f"Technical: {technical_score:.2f}, Fundamental: {fundamental_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Stage 2 module selection failed for {symbol}: {e}")
            return None
    
    def _momentum_module_analysis(self, symbol: str, forensic_analysis: Dict[str, Any], 
                                 fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                                 enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Momentum Module: Focus on strong momentum vs bull trap"""
        anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
        
        prompt = f"""
MOMENTUM MODULE ANALYSIS - Is this strong momentum or a bull trap?

=== DATA ===
{self._format_data_for_analysis(anonymized_data)}

=== FORENSIC ANALYSIS CONTEXT ===
{json.dumps(forensic_analysis, indent=2)}

=== MOMENTUM MODULE FOCUS ===

CRITICAL: RISK-REWARD CALCULATION INSTRUCTIONS
```
Given:
- Current Price: ₹{anonymized_data.get('technical', {}).get('current_price', anonymized_data.get('technical', {}).get('close_price', 'N/A'))}
- Resistance Level: [Calculate from 52-week high or technical resistance]
- Support Level: [Calculate from VWAP, 20-day SMA, or recent low]

Calculate:
1. Upside = (Resistance - Current Price)
2. Downside = (Current Price - Support)
3. Risk-Reward Ratio = Upside / Downside

Example:
- Current: ₹871
- Resistance: ₹880 (52w high)
- Support: ₹843 (20-day SMA)
- Upside: ₹880 - ₹871 = ₹9 (1.0%)
- Downside: ₹871 - ₹843 = ₹28 (3.2%)
- Ratio: 9/28 = 0.32:1 (means risk ₹28 to make ₹9)

INTERPRETATION:
- Ratio > 2.0:1 = Excellent (risk ₹1 to make ₹2+)
- Ratio 1.5-2.0:1 = Good
- Ratio 1.0-1.5:1 = Acceptable with tight stops
- Ratio < 1.0:1 = Poor setup, avoid or wait for better entry

YOU MUST:
- Show your calculation step by step
- Express ratio as X:1 (e.g., "2.5:1" means risk ₹1 to make ₹2.5)
- NEVER express as inverse (22:1 is WRONG if you mean 1:22)
```

Analyze specifically for momentum quality:

1. VOLUME CONFIRMATION:
   - Is volume supporting the price move? (Yes/No)
   - Volume vs 20-day average: [ratio]
   - Institutional volume patterns: [analysis]

2. TIMEFRAME ALIGNMENT:
   - Which timeframes show momentum? [list]
   - Are shorter timeframes confirming longer ones? (Yes/No)
   - Momentum consistency score: [0-1]

3. RSI MOMENTUM ANALYSIS:
   - RSI level: [value]
   - RSI trend: [rising/falling/sideways]
   - RSI interpretation: [Use RSI rules: >85=EXTREME, 80-85=Very overbought, 70-80=Overbought but manageable, 50-70=Bullish momentum, 30-50=Neutral, 20-30=Oversold, <20=EXTREME oversold]
   - RSI divergence: [present/absent]
   - Risk level from RSI: [HIGH/MODERATE/LOW/NO RISK based on RSI level]

4. MOMENTUM SUSTAINABILITY:
   - Can this momentum continue? (Yes/No)
   - Key factors supporting continuation: [list]
   - Potential momentum killers: [list]

5. BULL TRAP INDICATORS:
   - False breakout risk: [low/medium/high]
   - Volume divergence: [present/absent]
   - Resistance rejection: [present/absent]

6. MOMENTUM QUALITY SCORE (0-1):
   - Score: [0.0-1.0]
   - Reasoning: [detailed analysis]

Return as JSON:
{{
    "volume_confirmation": {{"supporting": true, "ratio": 0.0, "institutional_patterns": "text"}},
    "timeframe_alignment": {{"momentum_timeframes": ["list"], "confirmation": true, "consistency_score": 0.0}},
    "rsi_analysis": {{"level": 0, "trend": "rising", "interpretation": "text", "risk_level": "LOW", "divergence": "absent"}},
    "momentum_sustainability": {{"can_continue": true, "supporting_factors": ["list"], "momentum_killers": ["list"]}},
    "bull_trap_indicators": {{"false_breakout_risk": "low", "volume_divergence": false, "resistance_rejection": false}},
    "risk_reward_calculation": {{
        "current_price": 0.0,
        "resistance_level": 0.0,
        "support_level": 0.0,
        "upside": 0.0,
        "downside": 0.0,
        "upside_percentage": 0.0,
        "downside_percentage": 0.0,
        "risk_reward_ratio": 0.0,
        "ratio_interpretation": "text",
        "calculation_steps": "step by step calculation"
    }},
    "momentum_quality_score": {{"score": 0.0, "reasoning": "text"}},
    "module_conclusion": "strong_momentum" | "bull_trap" | "unclear"
}}
"""
        
        response = self._call_ai_provider(prompt, "You are a momentum trading specialist analyzing whether current price action represents genuine momentum or a potential bull trap.")
        if not response:
            return None
            
        return self._parse_json_response(response, "Momentum Module Analysis")
    
    def _value_entry_module_analysis(self, symbol: str, forensic_analysis: Dict[str, Any], 
                                   fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                                   enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Value Entry Module: Focus on why buy NOW vs waiting"""
        anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
        
        prompt = f"""
VALUE ENTRY MODULE ANALYSIS - Why buy NOW vs waiting?

=== DATA ===
{self._format_data_for_analysis(anonymized_data)}

=== FORENSIC ANALYSIS CONTEXT ===
{json.dumps(forensic_analysis, indent=2)}

=== VALUE ENTRY MODULE FOCUS ===

CRITICAL: RISK-REWARD CALCULATION INSTRUCTIONS
```
Given:
- Current Price: ₹{anonymized_data.get('technical', {}).get('current_price', anonymized_data.get('technical', {}).get('close_price', 'N/A'))}
- Resistance Level: [Calculate from 52-week high or technical resistance]
- Support Level: [Calculate from VWAP, 20-day SMA, or recent low]

Calculate:
1. Upside = (Resistance - Current Price)
2. Downside = (Current Price - Support)
3. Risk-Reward Ratio = Upside / Downside

Example:
- Current: ₹871
- Resistance: ₹880 (52w high)
- Support: ₹843 (20-day SMA)
- Upside: ₹880 - ₹871 = ₹9 (1.0%)
- Downside: ₹871 - ₹843 = ₹28 (3.2%)
- Ratio: 9/28 = 0.32:1 (means risk ₹28 to make ₹9)

INTERPRETATION:
- Ratio > 2.0:1 = Excellent (risk ₹1 to make ₹2+)
- Ratio 1.5-2.0:1 = Good
- Ratio 1.0-1.5:1 = Acceptable with tight stops
- Ratio < 1.0:1 = Poor setup, avoid or wait for better entry

YOU MUST:
- Show your calculation step by step
- Express ratio as X:1 (e.g., "2.5:1" means risk ₹1 to make ₹2.5)
- NEVER express as inverse (22:1 is WRONG if you mean 1:22)
```

Analyze specifically for value entry timing:

1. CATALYST ANALYSIS:
   - Recent earnings beat: [Yes/No, details]
   - Positive guidance revisions: [Yes/No, details]
   - Analyst upgrades: [Yes/No, details]
   - Institutional buying: [Yes/No, details]

2. VALUATION ATTRACTION:
   - Current P/E vs historical: [comparison]
   - P/B vs industry: [comparison]
   - PEG ratio: [value and interpretation]
   - Dividend yield: [value and sustainability]

3. ENTRY TIMING FACTORS:
   - Technical setup for entry: [good/poor/neutral]
   - Support level proximity: [close/far]
   - Risk-reward ratio: [favorable/unfavorable]
   - Market cycle position: [early/mid/late]

4. FUNDAMENTAL STRENGTHS:
   - Quality metrics: [strong/moderate/weak]
   - Growth prospects: [strong/moderate/weak]
   - Financial health: [strong/moderate/weak]
   - Competitive position: [strong/moderate/weak]

5. URGENCY FACTORS:
   - Why not wait for better entry? [analysis]
   - What could make it more expensive? [factors]
   - Downside protection: [level and reasoning]

6. VALUE ENTRY SCORE (0-1):
   - Score: [0.0-1.0]
   - Reasoning: [detailed analysis]

Return as JSON:
{{
    "catalyst_analysis": {{"earnings_beat": {{"present": true, "details": "text"}}, "guidance_revisions": {{"present": true, "details": "text"}}, "analyst_upgrades": {{"present": true, "details": "text"}}, "institutional_buying": {{"present": true, "details": "text"}}}},
    "valuation_attraction": {{"pe_vs_historical": "text", "pb_vs_industry": "text", "peg_ratio": {{"value": 0.0, "interpretation": "text"}}, "dividend_yield": {{"value": 0.0, "sustainability": "text"}}}},
    "entry_timing": {{"technical_setup": "good", "support_proximity": "close", "risk_reward": "favorable", "market_cycle": "early"}},
    "fundamental_strengths": {{"quality": "strong", "growth": "strong", "financial_health": "strong", "competitive_position": "strong"}},
    "urgency_factors": {{"why_not_wait": "text", "cost_increase_risks": ["list"], "downside_protection": {{"level": 0.0, "reasoning": "text"}}}},
    "risk_reward_calculation": {{
        "current_price": 0.0,
        "resistance_level": 0.0,
        "support_level": 0.0,
        "upside": 0.0,
        "downside": 0.0,
        "upside_percentage": 0.0,
        "downside_percentage": 0.0,
        "risk_reward_ratio": 0.0,
        "ratio_interpretation": "text",
        "calculation_steps": "step by step calculation"
    }},
    "value_entry_score": {{"score": 0.0, "reasoning": "text"}},
    "module_conclusion": "strong_value_entry" | "wait_for_better_entry" | "avoid_value_trap"
}}
"""
        
        response = self._call_ai_provider(prompt, "You are a value investing specialist analyzing the optimal entry timing for fundamentally strong stocks.")
        if not response:
            return None
            
        return self._parse_json_response(response, "Value Entry Module Analysis")
    
    def _balanced_module_analysis(self, symbol: str, forensic_analysis: Dict[str, Any], 
                                 fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                                 enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Balanced Module: Focus on which factors dominate in mixed signals"""
        anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
        
        prompt = f"""
BALANCED MODULE ANALYSIS - Which factors dominate in mixed signals?

=== DATA ===
{self._format_data_for_analysis(anonymized_data)}

=== FORENSIC ANALYSIS CONTEXT ===
{json.dumps(forensic_analysis, indent=2)}

=== BALANCED MODULE FOCUS ===

CRITICAL: RISK-REWARD CALCULATION INSTRUCTIONS
```
Given:
- Current Price: ₹{anonymized_data.get('technical', {}).get('current_price', anonymized_data.get('technical', {}).get('close_price', 'N/A'))}
- Resistance Level: [Calculate from 52-week high or technical resistance]
- Support Level: [Calculate from VWAP, 20-day SMA, or recent low]

Calculate:
1. Upside = (Resistance - Current Price)
2. Downside = (Current Price - Support)
3. Risk-Reward Ratio = Upside / Downside

Example:
- Current: ₹871
- Resistance: ₹880 (52w high)
- Support: ₹843 (20-day SMA)
- Upside: ₹880 - ₹871 = ₹9 (1.0%)
- Downside: ₹871 - ₹843 = ₹28 (3.2%)
- Ratio: 9/28 = 0.32:1 (means risk ₹28 to make ₹9)

INTERPRETATION:
- Ratio > 2.0:1 = Excellent (risk ₹1 to make ₹2+)
- Ratio 1.5-2.0:1 = Good
- Ratio 1.0-1.5:1 = Acceptable with tight stops
- Ratio < 1.0:1 = Poor setup, avoid or wait for better entry

YOU MUST:
- Show your calculation step by step
- Express ratio as X:1 (e.g., "2.5:1" means risk ₹1 to make ₹2.5)
- NEVER express as inverse (22:1 is WRONG if you mean 1:22)
```

Analyze mixed signals and determine dominance:

1. FACTOR WEIGHTING:
   - Fundamental strength: [0-1] - [reasoning]
   - Technical momentum: [0-1] - [reasoning]
   - Volume confirmation: [0-1] - [reasoning]
   - Risk factors: [0-1] - [reasoning]

2. CONFLICTING SIGNALS:
   - What signals are bullish? [list]
   - What signals are bearish? [list]
   - Which side has stronger evidence? [bullish/bearish/neutral]
   - Key deciding factors: [list]

3. RISK-REWARD ANALYSIS:
   - Upside potential: [percentage and reasoning]
   - Downside risk: [percentage and reasoning]
   - Risk-reward ratio: [ratio and interpretation]
   - Position sizing recommendation: [percentage]

4. TIMING CONSIDERATIONS:
   - Is this a good entry point? (Yes/No/Maybe)
   - What would make it better? [conditions]
   - What would make it worse? [conditions]
   - Optimal entry strategy: [strategy]

5. DOMINANT FACTORS:
   - Primary driver: [factor]
   - Secondary factors: [list]
   - Weakest link: [factor]
   - Strongest edge: [factor]

6. BALANCED SCORE (0-1):
   - Score: [0.0-1.0]
   - Reasoning: [detailed analysis]

Return as JSON:
{{
    "factor_weighting": {{"fundamental": {{"score": 0.0, "reasoning": "text"}}, "technical": {{"score": 0.0, "reasoning": "text"}}, "volume": {{"score": 0.0, "reasoning": "text"}}, "risk": {{"score": 0.0, "reasoning": "text"}}}},
    "conflicting_signals": {{"bullish": ["list"], "bearish": ["list"], "stronger_side": "bullish", "deciding_factors": ["list"]}},
    "risk_reward_calculation": {{
        "current_price": 0.0,
        "resistance_level": 0.0,
        "support_level": 0.0,
        "upside": 0.0,
        "downside": 0.0,
        "upside_percentage": 0.0,
        "downside_percentage": 0.0,
        "risk_reward_ratio": 0.0,
        "ratio_interpretation": "text",
        "calculation_steps": "step by step calculation"
    }},
    "timing": {{"good_entry": true, "improvement_conditions": ["list"], "worsening_conditions": ["list"], "entry_strategy": "text"}},
    "dominant_factors": {{"primary": "factor", "secondary": ["list"], "weakest_link": "factor", "strongest_edge": "factor"}},
    "balanced_score": {{"score": 0.0, "reasoning": "text"}},
    "module_conclusion": "bullish_dominance" | "bearish_dominance" | "neutral_wait"
}}
"""
        
        response = self._call_ai_provider(prompt, "You are a balanced analysis specialist who excels at weighing conflicting signals and determining which factors dominate in complex market situations.")
        if not response:
            return None
            
        return self._parse_json_response(response, "Balanced Module Analysis")
    
    def _stage3_risk_assessment(self, symbol: str, forensic_analysis: Dict[str, Any], 
                               module_analysis: Dict[str, Any], fundamentals: Dict[str, Any], 
                               technical: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Stage 3: Risk Assessment Prompt
        Identifies deal-breaker risks, manageable risks, position sizing, and risk level
        """
        try:
            # Get RSI value for the prompt
            rsi_value = technical.get('rsi_14', technical.get('rsi14', 'N/A'))
            
            prompt = f"""
RISK ASSESSMENT ANALYSIS - Identify all risk factors and determine position sizing

=== FORENSIC ANALYSIS CONTEXT ===
{json.dumps(forensic_analysis, indent=2)}

=== MODULE ANALYSIS CONTEXT ===
{json.dumps(module_analysis, indent=2)}

=== RISK ASSESSMENT TASK ===

RSI INTERPRETATION RULES:

RSI Level | Status | Action
----------|--------|--------
> 85 | EXTREME overbought | HIGH RISK - Avoid or very small position
80-85 | Very overbought | MODERATE RISK - Reduce position size 50%
70-80 | Overbought but manageable | LOW RISK - Normal position with trailing stop
50-70 | Bullish momentum | NO RISK - This is healthy uptrend
30-50 | Neutral | Assess other factors
20-30 | Oversold but manageable | Potential buy opportunity
< 20 | EXTREME oversold | Potential strong buy

Current RSI: {rsi_value}
Risk Level: [Calculate based on above table]

IMPORTANT: RSI 70-80 is NOT a deal-breaker in strong uptrends!
Only RSI > 85 should be considered HIGH RISK.

Analyze all risk factors and categorize them:

1. DEAL-BREAKER RISKS (must avoid):
   - Risk 1: [description and impact]
   - Risk 2: [description and impact]
   - Risk 3: [description and impact]

2. MANAGEABLE RISKS (can handle with stops):
   - Risk 1: [description and mitigation]
   - Risk 2: [description and mitigation]
   - Risk 3: [description and mitigation]

3. POSITION SIZING RECOMMENDATION:
   - Recommended size: [100%/75%/50%/25%]
   - Reasoning: [detailed explanation]
   - Risk-adjusted sizing: [considering all factors]

4. RISK LEVEL ASSESSMENT:
   - Overall risk level: [low/moderate/high/critical]
   - Primary risk drivers: [list]
   - Risk mitigation strategies: [list]

5. STOP-LOSS ANALYSIS:
   - Technical stop level: [price and reasoning]
   - Fundamental stop level: [price and reasoning]
   - Recommended stop: [price and reasoning]

6. RISK MONITORING:
   - Key risk indicators to watch: [list]
   - Risk escalation triggers: [list]
   - Exit conditions: [list]

Return as JSON:
{{
    "deal_breaker_risks": [{{"description": "text", "impact": "text"}}],
    "manageable_risks": [{{"description": "text", "mitigation": "text"}}],
    "position_sizing": {{"recommended_size": "50%", "reasoning": "text", "risk_adjusted": "text"}},
    "risk_level": "moderate",
    "primary_risk_drivers": ["list"],
    "risk_mitigation": ["list"],
    "stop_loss": {{"technical": {{"price": 0.0, "reasoning": "text"}}, "fundamental": {{"price": 0.0, "reasoning": "text"}}, "recommended": {{"price": 0.0, "reasoning": "text"}}}},
    "risk_monitoring": {{"key_indicators": ["list"], "escalation_triggers": ["list"], "exit_conditions": ["list"]}},
    "rsi_risk_assessment": {{
        "rsi_level": 0.0,
        "rsi_interpretation": "text",
        "risk_level": "LOW",
        "position_impact": "text"
    }}
}}
"""
            
            response = self._call_ai_provider(prompt, "You are a risk management specialist who excels at identifying, categorizing, and mitigating investment risks.")
            if not response:
                return None
                
            return self._parse_json_response(response, "Stage 3 Risk Assessment")
            
        except Exception as e:
            logger.error(f"Stage 3 risk assessment failed for {symbol}: {e}")
            return None
    
    def _stage4_final_decision(self, symbol: str, forensic_analysis: Dict[str, Any], 
                              module_analysis: Dict[str, Any], risk_assessment: Dict[str, Any], 
                              fundamentals: Dict[str, Any], technical: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Stage 4: Final Decision Prompt
        Forces binary decision based on combined analysis and risk assessment
        """
        try:
            # Calculate combined score
            forensic_score = forensic_analysis.get("overall_signal_strength", 0.0)
            module_score = module_analysis.get("module_analysis", {}).get("momentum_quality_score", {}).get("score", 0.0)
            if not module_score:
                module_score = module_analysis.get("module_analysis", {}).get("value_entry_score", {}).get("score", 0.0)
            if not module_score:
                module_score = module_analysis.get("module_analysis", {}).get("balanced_score", {}).get("score", 0.0)
            
            combined_score = (forensic_score + module_score) / 2
            
            # Get technical and fundamental scores
            tech_score = forensic_analysis.get("technical_score", 0.0)
            fund_score = forensic_analysis.get("fundamental_score", 0.0)
            
            # Check for deal-breakers
            deal_breakers = risk_assessment.get("deal_breaker_risks", [])
            has_deal_breakers = len(deal_breakers) > 0
            deal_breakers_text = f"{len(deal_breakers)} deal-breakers" if has_deal_breakers else "None"
            
            # Get risk level
            risk_level = risk_assessment.get("risk_level", "moderate")
            
            prompt = f"""
FINAL DECISION ANALYSIS - Force binary decision based on all previous analysis

=== COMBINED ANALYSIS SUMMARY ===
- Forensic Score: {forensic_score:.2f}
- Module Score: {module_score:.2f}
- Combined Score: {combined_score:.2f}
- Deal Breakers Present: {has_deal_breakers}

=== FORENSIC ANALYSIS ===
{json.dumps(forensic_analysis, indent=2)}

=== MODULE ANALYSIS ===
{json.dumps(module_analysis, indent=2)}

=== RISK ASSESSMENT ===
{json.dumps(risk_assessment, indent=2)}

=== FINAL DECISION RULES ===

MANDATORY DECISION MATRIX - FOLLOW EXACTLY:

IF combined_score >= 0.70 AND risk_level != "critical":
  → ACTION MUST BE: "buy" 
  → CONFIDENCE: 0.75-0.90
  → POSITION SIZE: 75-100%
  → YOU CANNOT SAY "AVOID" WITH SUCH STRONG SCORES!

ELSE IF combined_score >= 0.70 AND risk_level == "critical":
  → ACTION: "avoid" (only due to critical risk)
  → CONFIDENCE: 0.80-0.90

ELSE IF combined_score >= 0.65 AND risk_level == "moderate":
  → ACTION MUST BE: "buy" with reduced position (50-75%)
  → CONFIDENCE: 0.65-0.75
  → YOU CANNOT SAY "AVOID" WITH MODERATE RISK!

ELSE IF combined_score >= 0.65 AND risk_level == "high":
  → ACTION CAN BE: "buy" with 25-50% position OR "avoid"
  → Base decision on: Is there a clear edge despite risks?
  → CONFIDENCE: 0.55-0.70

ELSE IF combined_score >= 0.65 AND risk_level == "low":
  → ACTION MUST BE: "buy" with normal position (75-100%)
  → CONFIDENCE: 0.70-0.85
  → YOU CANNOT SAY "AVOID" WITH LOW RISK!

ELSE IF combined_score < 0.65:
  → ACTION: "avoid" or "watch"
  → CONFIDENCE: 0.50-0.70
  
ELSE IF deal_breakers exist (fraud, bankruptcy risk, regulatory ban):
  → ACTION: "avoid" regardless of score
  → CONFIDENCE: 0.80-0.95

Current Situation:
- Combined Score: {combined_score:.3f}
- Risk Level: {risk_level}
- Deal Breakers: {deal_breakers_text}

DECISION VALIDATION:
Based on the matrix above, your action SHOULD BE: [____]
If you are recommending "avoid" with combined_score >= 0.65 and risk_level != "critical", 
you are VIOLATING the decision matrix and being too conservative!

Now make your decision following this guidance EXACTLY.

CONFIDENCE SCORING GUIDE:

For "BUY" Actions:
- 0.85-1.0: All signals aligned, clear edge, low risk
- 0.75-0.84: Strong signals, minor conflicts, acceptable risk
- 0.65-0.74: Good setup but some concerns (use smaller position)
- 0.55-0.64: Marginal setup (only if forced to decide)
- < 0.55: Should not be buying

For "AVOID" Actions:
- 0.85-1.0: Clear deal-breakers, terrible setup
- 0.75-0.84: Multiple red flags, poor risk-reward
- 0.65-0.74: Concerns outweigh positives
- 0.55-0.64: Mixed but leaning negative
- < 0.55: Uncertain, better to wait

Current Case Analysis:
- Technical Score: {tech_score:.3f}
- Fundamental Score: {fund_score:.3f}
- Risk Level: {risk_level}
- Main Concerns: [list them]

If recommending AVOID with scores above 0.65, confidence should NOT be high.
If scores are strong but you're saying avoid due to minor concerns, 
you're being too conservative!

FORCED DECISION REQUIREMENTS:
1. BINARY CHOICE: You MUST choose one of: BUY, WATCH, or AVOID
2. NO DEFAULTING: Cannot default to "watch" without specific criteria
3. SPECIFIC TARGETS: Must provide stop loss, target price, holding period
4. CONVICTION: Must justify why this is the right decision

Return as JSON:
{{
    "action": "buy" | "watch" | "avoid",
    "confidence": 0.0-1.0,
    "rationale": "Comprehensive reasoning for the decision (max 200 words)",
    "position_size": "100%" | "75%" | "50%" | "25%",
    "stop_loss": {{"price": 0.0, "reasoning": "text"}},
    "target_price": {{"price": 0.0, "reasoning": "text"}},
    "holding_period": "short-term" | "medium-term" | "long-term",
    "entry_criteria": "Specific conditions for entry (if watch)",
    "decision_confidence": "high" | "medium" | "low",
    "key_factors": ["Top 3 factors driving this decision"],
    "risk_acknowledgment": "How risks are being managed",
    "decision_validation": {{
        "matrix_applied": true,
        "expected_action": "text based on matrix",
        "actual_action": "text",
        "matrix_compliance": true,
        "violation_check": "If recommending avoid with score >= 0.65 and risk != critical, this violates the matrix",
        "confidence_justification": "Why this confidence level is appropriate",
        "position_size_justification": "Why this position size is appropriate for the risk level"
    }}
}}
"""
            
            response = self._call_ai_provider(prompt, "You are a decisive investment manager who must make clear buy/sell/watch decisions based on comprehensive analysis. You cannot default to 'watch' without specific entry criteria.")
            if not response:
                return None
                
            return self._parse_json_response(response, "Stage 4 Final Decision")
            
        except Exception as e:
            logger.error(f"Stage 4 final decision failed for {symbol}: {e}")
            return None
    
    def _format_data_for_analysis(self, anonymized_data: Dict[str, Any]) -> str:
        """Format anonymized data for analysis prompts"""
        fundamentals = anonymized_data.get("fundamentals", {})
        technical = anonymized_data.get("technical", {})
        enhanced_fundamentals = anonymized_data.get("enhanced_fundamentals", {})
        
        formatted = "=== FUNDAMENTAL DATA ===\n"
        for key, value in fundamentals.items():
            formatted += f"- {key.upper()}: {value}\n"
        
        formatted += "\n=== TECHNICAL DATA ===\n"
        for key, value in technical.items():
            if value is not None:
                formatted += f"- {key.upper()}: {value}\n"
        
        if enhanced_fundamentals:
            formatted += "\n=== ENHANCED FUNDAMENTALS ===\n"
            for category, metrics in enhanced_fundamentals.items():
                if isinstance(metrics, dict):
                    formatted += f"\n{category.upper()}:\n"
                    for metric, value in metrics.items():
                        if value is not None:
                            formatted += f"- {metric}: {value}\n"
        
        return formatted
    
    def _call_ai_provider(self, prompt: str, system_message: str) -> Optional[str]:
        """Call the configured AI provider (OpenAI or Claude) with the given prompt"""
        try:
            if self.ai_provider.lower() == "claude":
                return self._call_claude(prompt, system_message)
            else:
                return self._call_openai(prompt, system_message)
                
        except Exception as e:
            logger.error(f"❌ MULTI-STAGE: AI provider call failed: {e}")
            return None
    
    def _call_openai(self, prompt: str, system_message: str) -> Optional[str]:
        """Call OpenAI API with the given prompt"""
        try:
            if not self.openai_client.client:
                logger.error("OpenAI client not available")
                return None
            
            response = self.openai_client.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent analysis
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ MULTI-STAGE: OpenAI API call failed: {e}")
            return None
    
    def _call_claude(self, prompt: str, system_message: str) -> Optional[str]:
        """Call Claude API with the given prompt"""
        try:
            if not self.claude_client.client:
                logger.error("Claude client not available")
                return None
            
            response = self.claude_client.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=0.1,  # Lower temperature for more consistent analysis
                system=system_message,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Check if response might be truncated
            if len(response_text) > 0.9 * settings.claude_max_tokens * 4:  # Rough estimate: 4 chars per token
                logger.warning(f"⚠️ Response length ({len(response_text)}) is close to max_tokens limit ({settings.claude_max_tokens})")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ MULTI-STAGE: Claude API call failed: {e}")
            return None
    
    def _parse_json_response(self, response, stage_name: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from AI provider with robust error handling"""
        try:
            # If response is already a dictionary, return it directly
            if isinstance(response, dict):
                logger.info(f"✅ {stage_name} received dictionary directly")
                return response
            
            # If response is a string, parse it as JSON
            if isinstance(response, str):
                # Try to find JSON in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
            
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    
                    # Clean up common JSON issues
                    json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                    json_str = json_str.replace('  ', ' ')  # Remove double spaces
                    
                    # Try to parse JSON
                    try:
                        parsed = json.loads(json_str)
                        logger.info(f"✅ {stage_name} parsed successfully")
                        return parsed
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"⚠️ First JSON parse failed, attempting to fix: {json_err}")
                        
                        # Check if response was truncated
                        if json_str.count('{') > json_str.count('}'):
                            logger.warning(f"⚠️ {stage_name} response appears truncated - missing closing braces")
                            # Try to complete the JSON by adding missing closing braces
                            missing_braces = json_str.count('{') - json_str.count('}')
                            json_str += '}' * missing_braces
                            logger.info(f"🔧 Added {missing_braces} closing braces to complete JSON")
                        
                        # Check for incomplete JSON objects (missing closing quotes, etc.)
                        if json_str.endswith(','):
                            json_str = json_str.rstrip(',') + '}'
                            logger.info(f"🔧 Removed trailing comma and added closing brace")
                        
                        # Fix common JSON issues
                        import re
                        # Fix unescaped quotes in strings
                        json_str = re.sub(r'([^\\])\"([^"]*)\"([^,}\]]*)\"', r'\1"\2\3"', json_str)
                        
                        try:
                            parsed = json.loads(json_str)
                            logger.info(f"✅ {stage_name} parsed successfully after fixing")
                            return parsed
                        except json.JSONDecodeError as final_err:
                            logger.error(f"❌ Failed to parse {stage_name} JSON: {final_err}")
                            return None
                else:
                    logger.error(f"❌ No JSON found in {stage_name} response")
                    return None
            else:
                logger.error(f"❌ Invalid response type for {stage_name}: {type(response)}")
                return None
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"❌ Failed to parse {stage_name} JSON: {e}")
            logger.error(f"Response content: {response[:1000]}...")
            return None

# Singleton instance
multi_stage_prompting_service = MultiStagePromptingService()
