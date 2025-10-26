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
from app.analysis.utilities.parse_json import _parse_json_response

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
        
        # Cache for processed data to avoid redundant processing
        self._processed_data_cache = {}
    
    def _prepare_data_once(self, symbol: str, fundamentals: Dict[str, Any], 
                          technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        OPTIMIZATION: Prepare all data once and reuse across all stages
        Eliminates redundant data anonymization and formatting
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{hash(str(fundamentals))}_{hash(str(technical))}"
            if cache_key in self._processed_data_cache:
                return self._processed_data_cache[cache_key]
            
            # Single data anonymization - COMMENTED OUT FOR TESTING
            # anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
            # Use raw data instead of anonymized data
            anonymized_data = {
                "fundamentals": fundamentals,
                "technical": technical,
                "enhanced_fundamentals": enhanced_fundamentals or {},
                "symbol": symbol  # Include symbol for better analysis
            }
            
            # Single data formatting
            formatted_data = self._format_data_for_analysis(anonymized_data)
            
            # Single risk-reward calculation with AI enhancement
            risk_reward = self._calculate_risk_reward_once(symbol, technical, fundamentals, enhanced_fundamentals)
            
            # Prepare comprehensive data package
            processed_data = {
                "anonymized_data": anonymized_data,
                "formatted_data": formatted_data,
                "risk_reward": risk_reward,
                "symbol": symbol,
                "fundamentals": fundamentals,
                "technical": technical,
                "enhanced_fundamentals": enhanced_fundamentals
            }
            
            # Cache the processed data
            self._processed_data_cache[cache_key] = processed_data
            
            # Limit cache size to prevent memory issues
            if len(self._processed_data_cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self._processed_data_cache))
                del self._processed_data_cache[oldest_key]
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare data for {symbol}: {e}")
            return None
    
    def _calculate_risk_reward_once(self, symbol: str, technical: Dict[str, Any], 
                                   fundamentals: Dict[str, Any] = None, enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ENHANCED: Calculate risk-reward with real money impacts and plain English translation
        """
        try:
            # Look for current_price in basic_indicators first, then fallback to direct technical data
            basic_indicators = technical.get('basic_indicators', {})
            
            current_price = (basic_indicators.get('current_price') or 
                           basic_indicators.get('close') or 
                           technical.get('current_price') or 
                           technical.get('close') or 0)
            
            if not current_price:
                logger.warning(f"🔍 MULTI_STAGE: No current price available for {symbol}")
                return {"error": "No current price available"}
            
            # Step 1: Core mathematical calculation (reliable)
            resistance = self._calculate_resistance_level(technical)
            support = self._calculate_support_level(technical)
            
            # Validate calculations
            if resistance <= current_price:
                logger.error(f"❌ CRITICAL: Resistance ({resistance}) <= Current Price ({current_price}) - This is impossible!")
                resistance = current_price * 1.05  # Force valid resistance
                logger.warning(f"🔧 FIXED: Set resistance to {resistance}")
            
            if support >= current_price:
                logger.error(f"❌ CRITICAL: Support ({support}) >= Current Price ({current_price}) - This is impossible!")
                support = current_price * 0.95  # Force valid support
                logger.warning(f"🔧 FIXED: Set support to {support}")
            
            upside = resistance - current_price
            downside = current_price - support
            ratio = upside / downside if downside > 0 else 0
            
            # Final validation
            if ratio < 0:
                logger.error(f"❌ CRITICAL: Negative risk-reward ratio ({ratio:.2f}) - This indicates calculation error!")
                logger.error(f"Debug: Current={current_price}, Resistance={resistance}, Support={support}")
                logger.error(f"Debug: Upside={upside}, Downside={downside}")
                # Force positive ratio
                ratio = max(0.1, ratio)  # Minimum 0.1:1 ratio
                logger.warning(f"🔧 FIXED: Set ratio to {ratio}")
            
            # Step 2: Calculate real money impacts (no AI cost)
            real_money_impacts = self._calculate_real_money_impacts(current_price, upside, downside, ratio)
            
            # Step 3: Identify top drivers (cost-optimized)
            top_drivers = self._identify_top_drivers(symbol, technical, fundamentals, enhanced_fundamentals)
            
            # Step 4: AI enhancement for context and interpretation (optimized prompt)
            ai_enhancement = self._get_ai_risk_reward_enhancement(
                symbol, current_price, resistance, support, upside, downside, ratio, 
                technical, fundamentals, enhanced_fundamentals
            )
            
            # Step 5: Combine reliable calculation with AI enhancement
            risk_reward_data = {
                "current_price": current_price,
                "resistance_level": resistance,
                "support_level": support,
                "upside": upside,
                "downside": downside,
                "upside_percentage": (upside / current_price * 100) if current_price > 0 else 0,
                "downside_percentage": (downside / current_price * 100) if current_price > 0 else 0,
                "risk_reward_ratio": ratio,
                "ratio_display": f"1:{ratio:.2f}" if ratio < 1 else f"{ratio:.2f}:1",
                "ratio_direction": "risk:reward",
                "ratio_interpretation": ai_enhancement.get("enhanced_interpretation", self._interpret_risk_reward_ratio(ratio)),
                "calculation_steps": f"Upside: {resistance} - {current_price} = {upside:.2f}, Downside: {current_price} - {support} = {downside:.2f}, Ratio: {upside:.2f} / {downside:.2f} = {ratio:.2f}:1",
                "ai_enhancement": ai_enhancement,
                # NEW: Real money impacts (no AI cost)
                "real_money_impacts": real_money_impacts,
                "plain_english_summary": self._create_plain_english_summary(ratio, real_money_impacts),
                # NEW: Top drivers analysis
                "top_drivers": top_drivers
            }
            
            return risk_reward_data
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate risk-reward for {symbol}: {e}")
            return {"error": str(e)}
    
    def _get_ai_risk_reward_enhancement(self, symbol: str, current_price: float, 
                                      resistance: float, support: float, upside: float, 
                                      downside: float, ratio: float, technical: Dict[str, Any],
                                      fundamentals: Dict[str, Any] = None, enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        COST-OPTIMIZED: AI enhancement with plain English translation and contradiction resolution
        """
        try:
            # Extract only key indicators to reduce token usage
            rsi = technical.get('rsi_14', technical.get('rsi14'))
            pe_ratio = fundamentals.get('pe_ratio', fundamentals.get('trailing_pe')) if fundamentals else None
            sector = fundamentals.get('sector', fundamentals.get('industry')) if fundamentals else None
            
            # COST-OPTIMIZED PROMPT - Much shorter and focused
            prompt = f"""RISK-REWARD ANALYSIS: ₹{current_price} → ₹{resistance} (upside ₹{upside:.0f}) vs ₹{support} (downside ₹{downside:.0f}) = {ratio:.2f}:1

Key Data: RSI {rsi}, P/E {pe_ratio}, Sector {sector}

Tasks:
1. Translate jargon: RSI {rsi} = ?, P/E {pe_ratio} = ?
2. Resolve contradictions: Why bullish if RSI high/P/E high?
3. Plain English: What this means for investors

Return JSON:
{{
    "jargon_translation": {{"rsi": "text", "pe": "text"}},
    "contradiction_resolution": "text",
                "enhanced_interpretation": "text",
                "quality_score": 0.0,
                "confidence_level": "high/medium/low"
}}"""
            
            response = self._call_ai_provider(prompt, "You are a financial educator who explains complex analysis in simple terms.")
            if not response:
                # Fallback to basic interpretation if AI fails
                return {
                    "enhanced_interpretation": self._interpret_risk_reward_ratio(ratio),
                    "quality_score": 0.5,
                    "confidence_level": "medium",
                    "jargon_translation": {"rsi": f"RSI {rsi} - momentum indicator", "pe": f"P/E {pe_ratio} - valuation metric"},
                    "contradiction_resolution": "No major contradictions detected"
                }
            
            return _parse_json_response(response, "Risk-Reward Enhancement")
            
        except Exception as e:
            logger.error(f"❌ AI enhancement failed for {symbol}: {e}")
            # Fallback to basic interpretation
            return {
                "enhanced_interpretation": self._interpret_risk_reward_ratio(ratio),
                "quality_score": 0.5,
                "confidence_level": "medium",
                "jargon_translation": {"rsi": f"RSI {rsi} - momentum indicator", "pe": f"P/E {pe_ratio} - valuation metric"},
                "contradiction_resolution": "No major contradictions detected"
            }
    
    def _calculate_resistance_level(self, technical: Dict[str, Any]) -> float:
        """Calculate resistance level from technical data with proper validation"""
        current_price = self._get_current_price(technical)
        
        # Get basic indicators from nested structure
        basic_indicators = technical.get('basic_indicators', {})
        
        
        # Try 52-week high first
        high_52w = basic_indicators.get('high_52w', technical.get('high_52w', technical.get('fifty_two_week_high')))
        if high_52w and high_52w > current_price:
            return float(high_52w)
        
        # Try recent high
        recent_high = basic_indicators.get('recent_high', technical.get('recent_high', technical.get('high')))
        if recent_high and recent_high > current_price:
            # Additional validation: reject obviously corrupted values
            # Resistance should be within reasonable range (not more than 200% above current price)
            if recent_high > current_price * 2.0:
                logger.warning(f"Rejecting corrupted recent_high: {recent_high} (more than 200% above current price {current_price})")
            else:
                return float(recent_high)
        
        # Try VWAP as resistance
        vwap = basic_indicators.get('vwap', technical.get('vwap'))
        if vwap and vwap > current_price:
            return float(vwap)
        
        # Try SMA 20 as resistance
        sma_20 = basic_indicators.get('sma_20', technical.get('sma_20'))
        if sma_20 and sma_20 > current_price:
            return float(sma_20)
        
        # Fallback to current price * 1.05 (5% upside assumption)
        resistance = current_price * 1.05
        logger.warning(f"Using fallback resistance calculation: {current_price} * 1.05 = {resistance}")
        return resistance
    
    def _calculate_support_level(self, technical: Dict[str, Any]) -> float:
        """Calculate support level from technical data with proper validation"""
        current_price = self._get_current_price(technical)
        
        # Get basic indicators from nested structure
        basic_indicators = technical.get('basic_indicators', {})
        
        
        # Try VWAP first
        vwap = basic_indicators.get('vwap', technical.get('vwap', technical.get('vwap_20')))
        if vwap and vwap < current_price:
            return float(vwap)
        
        # Try 20-day SMA
        sma_20 = basic_indicators.get('sma_20', technical.get('sma_20', technical.get('sma20')))
        if sma_20 and sma_20 < current_price:
            return float(sma_20)
        
        # Try recent low
        recent_low = basic_indicators.get('recent_low', technical.get('recent_low', technical.get('low')))
        if recent_low and recent_low < current_price and recent_low > 0:
            # Additional validation: reject obviously corrupted values
            # Support should be within reasonable range (not more than 10% below current price)
            if recent_low < current_price * 0.9:
                logger.warning(f"Rejecting corrupted recent_low: {recent_low} (more than 10% below current price {current_price})")
            else:
                return float(recent_low)
        
        # Fallback to current price * 0.95 (5% downside assumption)
        support = current_price * 0.95
        logger.warning(f"Using fallback support calculation: {current_price} * 0.95 = {support}")
        return support
    
    def _get_current_price(self, technical: Dict[str, Any]) -> float:
        """Extract current price from technical data with proper fallbacks"""
        # Look for current_price in basic_indicators first, then fallback to direct technical data
        basic_indicators = technical.get('basic_indicators', {})
        
        current_price = (basic_indicators.get('current_price') or 
                       basic_indicators.get('close') or 
                       technical.get('current_price') or 
                       technical.get('close') or 0)
        
        if not current_price or current_price <= 0:
            logger.error(f"❌ CRITICAL: Invalid current price: {current_price}")
            return 100.0  # Fallback price
        
        return float(current_price)
    
    def _interpret_risk_reward_ratio(self, ratio: float) -> str:
        """Interpret risk-reward ratio"""
        if ratio >= 2.0:
            return "Excellent (risk ₹1 to make ₹2+)"
        elif ratio >= 1.5:
            return "Good"
        elif ratio >= 1.0:
            return "Acceptable with tight stops"
        else:
            return "Poor setup, avoid or wait for better entry"
    
    def _calculate_real_money_impacts(self, current_price: float, upside: float, 
                                     downside: float, ratio: float) -> Dict[str, Any]:
        """Calculate real money impacts for different investment amounts (no AI cost)"""
        investment_amounts = [10000, 50000, 100000]  # ₹10K, ₹50K, ₹1L (reduced for cost)
        
        real_money_impacts = {}
        for amount in investment_amounts:
            shares = amount / current_price if current_price > 0 else 0
            
            # Calculate potential gains/losses
            upside_pct = upside / current_price if current_price > 0 else 0
            downside_pct = downside / current_price if current_price > 0 else 0
            
            potential_gain = amount * upside_pct
            potential_loss = amount * downside_pct
            
            real_money_impacts[f"₹{amount:,}"] = {
                "shares": f"{shares:.0f}",
                "potential_gain": f"₹{potential_gain:,.0f}",
                "potential_loss": f"₹{potential_loss:,.0f}",
                "net_risk_reward": f"Risk ₹{potential_loss:,.0f} to make ₹{potential_gain:,.0f}",
                "gain_pct": f"{upside_pct*100:.1f}%",
                "loss_pct": f"{downside_pct*100:.1f}%"
            }
        
        return real_money_impacts

    def _create_plain_english_summary(self, ratio: float, real_money_impacts: Dict[str, Any]) -> Dict[str, str]:
        """Create plain English summary (no AI cost)"""
        if ratio >= 2.0:
            risk_level = "Low Risk, High Reward"
            advice = "Good setup - risk ₹1 to make ₹2+"
            ratio_text = f"{ratio:.2f}:1 ratio"
        elif ratio >= 1.5:
            risk_level = "Moderate Risk, Good Reward"
            advice = "Decent setup - risk ₹1 to make ₹1.50"
            ratio_text = f"{ratio:.2f}:1 ratio"
        elif ratio >= 1.0:
            risk_level = "Balanced Risk-Reward"
            advice = "Fair setup - risk ₹1 to make ₹1"
            ratio_text = f"{ratio:.2f}:1 ratio"
        else:
            risk_level = "High Risk, Low Reward"
            advice = "Poor setup - avoid or wait for better entry"
            ratio_text = f"1:{ratio:.2f} ratio (risk ₹{1/ratio:.0f} to make ₹1)"
        
        # Get sample for context
        sample = list(real_money_impacts.keys())[0] if real_money_impacts else "₹10,000"
        sample_data = real_money_impacts.get(sample, {})
        
        return {
            "risk_level": risk_level,
            "advice": advice,
            "example": f"For {sample}: {sample_data.get('net_risk_reward', 'N/A')}",
            "ratio_meaning": ratio_text
        }
    
    def _identify_top_drivers(self, symbol: str, technical: Dict[str, Any], 
                             fundamentals: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        COST-OPTIMIZED: Identify top 5 drivers from all stored data that influence decision-making
        """
        try:
            # Extract key metrics only (reduce token usage)
            key_metrics = {
                "technical": {
                    "rsi": technical.get('rsi_14', technical.get('rsi14')),
                    "macd": technical.get('macd'),
                    "sma_20": technical.get('sma_20'),
                    "volume": technical.get('volume'),
                    "atr": technical.get('atr_14'),
                    "price_change_1d": technical.get('price_change_1d_pct', 0) * 100,
                    "price_change_5d": technical.get('price_change_5d_pct', 0) * 100
                },
                "fundamental": {
                    "pe_ratio": fundamentals.get('pe_ratio', fundamentals.get('trailing_pe')),
                    "pb_ratio": fundamentals.get('pb_ratio'),
                    "roe": fundamentals.get('roe'),
                    "debt_equity": fundamentals.get('debt_equity_ratio'),
                    "current_ratio": fundamentals.get('current_ratio'),
                    "revenue_growth": fundamentals.get('revenue_growth_yoy'),
                    "eps_growth": fundamentals.get('eps_growth_yoy')
                },
                "enhanced": {
                    "operating_margin": enhanced_fundamentals.get('quality_metrics', {}).get('operating_margin') if enhanced_fundamentals else None,
                    "free_cash_flow": enhanced_fundamentals.get('growth_metrics', {}).get('free_cash_flow_growth') if enhanced_fundamentals else None,
                    "dividend_yield": enhanced_fundamentals.get('value_metrics', {}).get('dividend_yield') if enhanced_fundamentals else None
                }
            }
            
            # COST-OPTIMIZED PROMPT - Very focused
            prompt = f"""TOP DRIVERS ANALYSIS: {symbol}

Key Metrics: {json.dumps(key_metrics, indent=1)}

Task: Identify top 5 drivers that most influence BUY/SELL decision.

Return JSON:
{{
    "drivers": [
        {{"metric": "RSI", "value": 47, "impact": "high/medium/low", "explanation": "why this matters"}},
        {{"metric": "P/E", "value": 25, "impact": "high/medium/low", "explanation": "why this matters"}},
        {{"metric": "Revenue Growth", "value": "15%", "impact": "high/medium/low", "explanation": "why this matters"}},
        {{"metric": "MACD", "value": "bullish", "impact": "high/medium/low", "explanation": "why this matters"}},
        {{"metric": "Debt/Equity", "value": 0.3, "impact": "high/medium/low", "explanation": "why this matters"}}
    ],
    "decision_influence": "Which 2-3 factors are most critical for this stock's decision",
    "risk_factors": ["Top 2-3 risk factors to watch"],
    "opportunity_factors": ["Top 2-3 opportunity factors"]
}}"""
            
            response = self._call_ai_provider(prompt, "You are a financial analyst who identifies the most important metrics that drive investment decisions.")
            if not response:
                # Fallback to basic driver identification
                return self._get_fallback_drivers(key_metrics)
            
            return _parse_json_response(response, "Top Drivers Analysis")
            
        except Exception as e:
            logger.error(f"❌ Top drivers analysis failed for {symbol}: {e}")
            return self._get_fallback_drivers({})
    
    def _get_fallback_drivers(self, key_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback driver identification when AI fails"""
        technical = key_metrics.get('technical', {})
        fundamental = key_metrics.get('fundamental', {})
        
        # Simple rule-based driver identification
        drivers = []
        
        # RSI driver
        rsi = technical.get('rsi')
        if rsi is not None:
            if rsi > 70:
                drivers.append({"metric": "RSI", "value": rsi, "impact": "high", "explanation": "Overbought - price may fall"})
            elif rsi < 30:
                drivers.append({"metric": "RSI", "value": rsi, "impact": "high", "explanation": "Oversold - potential bounce"})
            else:
                drivers.append({"metric": "RSI", "value": rsi, "impact": "medium", "explanation": "Neutral momentum"})
        
        # P/E driver
        pe = fundamental.get('pe_ratio')
        if pe is not None:
            if pe < 15:
                drivers.append({"metric": "P/E Ratio", "value": pe, "impact": "high", "explanation": "Undervalued - good buying opportunity"})
            elif pe > 30:
                drivers.append({"metric": "P/E Ratio", "value": pe, "impact": "high", "explanation": "Overvalued - expensive stock"})
            else:
                drivers.append({"metric": "P/E Ratio", "value": pe, "impact": "medium", "explanation": "Fairly valued"})
        
        # Revenue growth driver
        revenue_growth = fundamental.get('revenue_growth')
        if revenue_growth is not None:
            if revenue_growth > 20:
                drivers.append({"metric": "Revenue Growth", "value": f"{revenue_growth}%", "impact": "high", "explanation": "Strong growth - positive for stock"})
            elif revenue_growth < 0:
                drivers.append({"metric": "Revenue Growth", "value": f"{revenue_growth}%", "impact": "high", "explanation": "Declining revenue - negative signal"})
            else:
                drivers.append({"metric": "Revenue Growth", "value": f"{revenue_growth}%", "impact": "medium", "explanation": "Moderate growth"})
        
        return {
            "drivers": drivers[:5],  # Limit to 5
            "decision_influence": "Key metrics that drive investment decisions",
            "risk_factors": ["High P/E ratio", "Declining revenue"],
            "opportunity_factors": ["Low P/E ratio", "Strong revenue growth"]
        }
        
    
    def _stage1_simple_analysis(self, symbol: str, fundamentals: Dict[str, Any], 
                               technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None, processed_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        ENHANCED: Stage 1: Simple 3-Factor Analysis using pre-processed data
        Identifies the 3 most important factors: Setup, Catalyst, Confirmation
        """
        try:
            # OPTIMIZATION: Use pre-processed data if available
            if processed_data:
                anonymized_data = processed_data["anonymized_data"]
                formatted_data = processed_data["formatted_data"]
                risk_reward = processed_data["risk_reward"]
            else:
                # Fallback to original processing
                # anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
                # Use raw data instead of anonymized data
                anonymized_data = {
                    "fundamentals": fundamentals,
                    "technical": technical,
                    "enhanced_fundamentals": enhanced_fundamentals or {},
                    "symbol": symbol
                }
                formatted_data = self._format_data_for_analysis(anonymized_data)
                risk_reward = self._calculate_risk_reward_once(symbol, technical, fundamentals, enhanced_fundamentals)
            
            # COST-OPTIMIZED PROMPT - Focused on Setup/Catalyst/Confirmation only
            prompt = f"""STOCK ANALYSIS: {symbol}

Data: {formatted_data[:500]}...

Risk-Reward: {risk_reward.get('risk_reward_ratio', 0):.2f}:1 (₹{risk_reward.get('upside', 0):.0f} upside vs ₹{risk_reward.get('downside', 0):.0f} downside)
Real Money: {json.dumps(risk_reward.get('real_money_impacts', {}), indent=1)}

Tasks:
1. SETUP: What's happening with price/technicals in simple terms
2. CATALYST: Why this timing matters for investors  
3. CONFIRMATION: What confirms this is real opportunity

IMPORTANT: When mentioning technical terms, always explain their impact on decision-making:
- P/E ratio → "expensive/cheap relative to earnings"
- P/B ratio → "premium/discount to book value" 
- RSI → "overbought/oversold momentum"
- MACD → "trend direction signal"
- OBV → "volume buying/selling pressure"
- Operating margin → "profitability strength"

Focus on HOW IMPORTANT each factor is for the decision, not just what the numbers are.

LANGUAGE STYLE - Write in Simple, Everyday Language:
- Write as if explaining to a friend who doesn't understand finance jargon
- Use analogies and comparisons: "like a car that's both fuel-efficient AND fast"
- Instead of "RSI at 50.31" → say "the price momentum is neutral, not too hot or too cold"
- Instead of "P/E ratio 15.47x" → say "trading at 15 times earnings, which is reasonable (not too expensive)"
- Instead of "risk-reward ratio 0.17:1" → say "this is risky - you could lose ₹6 for every ₹1 you might make"
- Instead of "negative free cash flow -24.10M" → say "the company is spending more cash than it's bringing in, which is concerning"
- Instead of "OBV -302,609" → say "more people are selling than buying right now"
- Use "good/bad", "cheap/expensive", "strong/weak" instead of technical terms
- Focus on practical impact: "This could make you ₹X if it goes well, or lose you ₹Y if it doesn't"

TRANSLATE JARGON TO PLAIN LANGUAGE:
- "Risk-reward ratio 0.17:1" → "Poor odds - you could lose ₹6 for every ₹1 you might gain"
- "P/E 27.4x" → "Expensive - paying 27 times what the company earns per share"
- "OBV -302,609" → "Heavy selling pressure - more people are selling than buying"
- "RSI 46.8" → "Neutral momentum - stock is neither overbought nor oversold"
- "Negative free cash flow -$24M" → "Company is burning cash - spending more than it earns"
- "High debt-to-equity" → "Company has a lot of debt compared to its worth"
- "Price above SMA 20" → "Currently trading above its recent average price"

Return JSON with scores (0.0-1.0):
{{
    "the_setup": "text",
    "the_catalyst": "text", 
    "the_confirmation": "text",
    "overall_signal": "strong/weak/neutral",
    "confidence": 0.0,
    "fundamental_score": 0.0,  // Score 0.0-1.0 based on valuation, growth, profitability
    "technical_score": 0.0,   // Score 0.0-1.0 based on momentum, trends, indicators
    "overall_signal_strength": 0.0,
    "primary_driver": "momentum"
}}

SCORING GUIDELINES:
Technical Score:
- 0.7-1.0: Very bullish (strong uptrend, positive momentum, volume confirmation)
- 0.5-0.7: Moderately bullish (mixed signals, slight advantage bullish)
- 0.3-0.5: Weak/neutral (mixed signals, no clear direction)
- 0.0-0.3: Bearish (weak technicals, negative momentum)

Fundamental Score:
- 0.7-1.0: Very attractive (good valuation, strong growth, healthy metrics)
- 0.5-0.7: Reasonable fundamentals (fair value, decent prospects)
- 0.3-0.5: Weak fundamentals (overvalued or poor quality)
- 0.0-0.3: Very weak fundamentals (expensive, declining)

Confidence: Your conviction in this analysis (0.0-1.0)
}}"""
            
            response = self._call_ai_provider(prompt, "You are a financial advisor who explains stock investments in simple, everyday language. Avoid jargon. Use analogies and real-world examples. Write as if explaining to a friend who isn't a finance expert. Focus on what matters for making money, not technical complexity.")
            if not response:
                return None
            result = _parse_json_response(response, "Stage 1 Simple Analysis")
            result["risk_reward"] = risk_reward
            return result
            
        except Exception as e:
            logger.error(f"Stage 1 analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def _stage2_simple_decision(self, symbol: str, simple_analysis: Dict[str, Any], 
                               fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                               enhanced_fundamentals: Dict[str, Any] = None, processed_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        ENHANCED: Stage 2: Simple Decision using simple analysis output as input
        Replaces Stages 2, 3, 4 with single focused decision
        """
        try:
            # Use simple analysis result as input
            the_setup = simple_analysis.get("the_setup", "")
            the_catalyst = simple_analysis.get("the_catalyst", "")
            the_confirmation = simple_analysis.get("the_confirmation", "")
            overall_signal = simple_analysis.get("overall_signal", "neutral")
            confidence = simple_analysis.get("confidence", 0.0)
            
            # ENHANCED PROMPT with clear decision criteria
            prompt = f"""DECISION: {symbol}

Setup: {the_setup}
Catalyst: {the_catalyst}  
Confirmation: {the_confirmation}
Signal: {overall_signal} (confidence: {confidence})

LANGUAGE REQUIREMENTS:
- Explain reasoning in plain English, not finance jargon
- If you must mention technical terms, immediately explain what they mean for the investor
- Use simple comparisons: "like a stock that's been climbing" instead of "showing upward momentum"
- Write bullets as clear, actionable insights
- Focus on what will happen to the investor's money, not abstract metrics

DECISION CRITERIA:
- BUY: Strong setup + clear catalyst + good confirmation (strong signal, confidence > 0.6)
- AVOID: Weak fundamentals + poor technicals + high risk (weak signal, confidence > 0.7)
- WATCH: Mixed signals, wait for clarity, or borderline confidence

Make a clear, confident decision. Prefer BUY or AVOID over WATCH when signals are clear.

Return JSON:
            {{
                "decision": "BUY/WATCH/AVOID",
                "confidence": 0.0,
                "reasoning": ["bullet1", "bullet2", "bullet3"],
    "position_size": "50%",
                "stop_loss": 0.0,
                "target_price": 0.0,
    "risk_level": "low/moderate/high"
}}"""
            
            response = self._call_ai_provider(prompt, "You are a financial advisor who makes clear investment decisions and explains them in simple language that anyone can understand.")
            if not response:
                return None
                
            return _parse_json_response(response, "Stage 2 Simple Decision")
            
        except Exception as e:
            logger.error(f"Stage 2 simple decision failed for {symbol}: {e}")
            return None
    

            return None
    
    def _format_data_for_analysis(self, anonymized_data: Dict[str, Any]) -> str:
        """Format anonymized data for analysis prompts with proper flattening of nested structures"""
        fundamentals = anonymized_data.get("fundamentals", {})
        technical = anonymized_data.get("technical", {})
        enhanced_fundamentals = anonymized_data.get("enhanced_fundamentals", {})
        
        formatted = "=== FUNDAMENTAL DATA ===\n"
        for key, value in fundamentals.items():
            formatted += f"- {key.upper()}: {value}\n"
        
        formatted += "\n=== TECHNICAL DATA ===\n"
        # FIX: Flatten nested technical indicators so AI can see individual values
        for key, value in technical.items():
            if value is not None:
                if isinstance(value, dict):
                    # Flatten nested dictionaries (basic_indicators, momentum_indicators, etc.)
                    for indicator_name, indicator_value in value.items():
                        if indicator_value is not None:
                            # Skip None values and boolean indicators
                            if not isinstance(indicator_value, bool):
                                formatted += f"- {indicator_name.upper()}: {indicator_value}\n"
                else:
                    # Keep simple values as-is
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
                temperature=0.3,  # Higher temperature for more natural, conversational language
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
                temperature=0.3,  # Higher temperature for more natural, conversational language
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
    
    
# Singleton instance
multi_stage_prompting_service = MultiStagePromptingService()
