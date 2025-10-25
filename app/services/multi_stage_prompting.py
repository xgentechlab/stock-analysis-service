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
                logger.debug(f"🎯 Using cached processed data for {symbol}")
                return self._processed_data_cache[cache_key]
            
            logger.debug(f"📊 Processing data once for {symbol}")
            
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
            
            logger.debug(f"✅ Data processed and cached for {symbol}")
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
            logger.info(f"🔍 MULTI_STAGE: Received technical data keys: {list(technical.keys())}")
            logger.info(f"🔍 MULTI_STAGE: Full technical data structure:")
            for key, value in technical.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {list(value.keys()) if value else 'empty dict'}")
                else:
                    logger.info(f"  {key}: {value}")
            
            logger.info(f"🔍 MULTI_STAGE: technical.get('basic_indicators') = {technical.get('basic_indicators')}")
            logger.info(f"🔍 MULTI_STAGE: technical.get('current_price') = {technical.get('current_price')}")
            logger.info(f"🔍 MULTI_STAGE: technical.get('close') = {technical.get('close')}")
            
            # Look for current_price in basic_indicators first, then fallback to direct technical data
            basic_indicators = technical.get('basic_indicators', {})
            logger.info(f"🔍 MULTI_STAGE: basic_indicators = {basic_indicators}")
            
            current_price = (basic_indicators.get('current_price') or 
                           basic_indicators.get('close') or 
                           technical.get('current_price') or 
                           technical.get('close') or 0)
            
            logger.info(f"🔍 MULTI_STAGE: Final current_price = {current_price}")
            
            if not current_price:
                logger.warning(f"🔍 MULTI_STAGE: No current price available for {symbol}")
                return {"error": "No current price available"}
            
            # Step 1: Core mathematical calculation (reliable)
            logger.info(f"🔍 MAIN CALC: Starting risk-reward calculation for {symbol}")
            logger.info(f"🔍 MAIN CALC: Current price = {current_price}")
            
            resistance = self._calculate_resistance_level(technical)
            support = self._calculate_support_level(technical)
            
            logger.info(f"🔍 MAIN CALC: Calculated resistance = {resistance}")
            logger.info(f"🔍 MAIN CALC: Calculated support = {support}")
            
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
            
            logger.info(f"🔍 MAIN CALC: Final values - Upside: {upside}, Downside: {downside}, Ratio: {ratio}")
            
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
                "ratio_interpretation": ai_enhancement.get("enhanced_interpretation", self._interpret_risk_reward_ratio(ratio)),
                "calculation_steps": f"Upside: {resistance} - {current_price} = {upside:.2f}, Downside: {current_price} - {support} = {downside:.2f}, Ratio: {upside:.2f} / {downside:.2f} = {ratio:.2f}:1",
                "ai_enhancement": ai_enhancement,
                # NEW: Real money impacts (no AI cost)
                "real_money_impacts": real_money_impacts,
                "plain_english_summary": self._create_plain_english_summary(ratio, real_money_impacts),
                # NEW: Top drivers analysis
                "top_drivers": top_drivers
            }
            
            logger.debug(f"🎯 Risk-reward calculated for {symbol}: {ratio:.2f}:1 (AI enhanced)")
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
            
            return self._parse_json_response(response, "Risk-Reward Enhancement")
            
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
        
        logger.info(f"🔍 RESISTANCE CALC: Current price = {current_price}")
        logger.info(f"🔍 RESISTANCE CALC: Technical data keys = {list(technical.keys())}")
        
        # Get basic indicators from nested structure
        basic_indicators = technical.get('basic_indicators', {})
        logger.info(f"🔍 RESISTANCE CALC: basic_indicators = {basic_indicators}")
        
        # Try 52-week high first
        high_52w = basic_indicators.get('high_52w', technical.get('high_52w', technical.get('fifty_two_week_high')))
        logger.info(f"🔍 RESISTANCE CALC: high_52w = {high_52w}")
        if high_52w and high_52w > current_price:
            logger.info(f"✅ RESISTANCE CALC: Using high_52w = {high_52w}")
            return float(high_52w)
        else:
            logger.info(f"❌ RESISTANCE CALC: high_52w rejected (value={high_52w}, condition={high_52w and high_52w > current_price})")
        
        # Try recent high
        recent_high = basic_indicators.get('recent_high', technical.get('recent_high', technical.get('high')))
        logger.info(f"🔍 RESISTANCE CALC: recent_high = {recent_high}")
        if recent_high and recent_high > current_price:
            # Additional validation: reject obviously corrupted values
            # Resistance should be within reasonable range (not more than 200% above current price)
            if recent_high > current_price * 2.0:
                logger.warning(f"❌ RESISTANCE CALC: Rejecting corrupted recent_high: {recent_high} (more than 200% above current price {current_price})")
            else:
                logger.info(f"✅ RESISTANCE CALC: Using recent_high = {recent_high}")
                return float(recent_high)
        else:
            logger.info(f"❌ RESISTANCE CALC: recent_high rejected (value={recent_high}, condition={recent_high and recent_high > current_price})")
        
        # Try VWAP as resistance
        vwap = basic_indicators.get('vwap', technical.get('vwap'))
        logger.info(f"🔍 RESISTANCE CALC: vwap = {vwap}")
        if vwap and vwap > current_price:
            logger.info(f"✅ RESISTANCE CALC: Using vwap = {vwap}")
            return float(vwap)
        else:
            logger.info(f"❌ RESISTANCE CALC: vwap rejected (value={vwap}, condition={vwap and vwap > current_price})")
        
        # Try SMA 20 as resistance
        sma_20 = basic_indicators.get('sma_20', technical.get('sma_20'))
        logger.info(f"🔍 RESISTANCE CALC: sma_20 = {sma_20}")
        if sma_20 and sma_20 > current_price:
            logger.info(f"✅ RESISTANCE CALC: Using sma_20 = {sma_20}")
            return float(sma_20)
        else:
            logger.info(f"❌ RESISTANCE CALC: sma_20 rejected (value={sma_20}, condition={sma_20 and sma_20 > current_price})")
        
        # Fallback to current price * 1.05 (5% upside assumption)
        resistance = current_price * 1.05
        logger.warning(f"⚠️ RESISTANCE CALC: Using fallback calculation: {current_price} * 1.05 = {resistance}")
        return resistance
    
    def _calculate_support_level(self, technical: Dict[str, Any]) -> float:
        """Calculate support level from technical data with proper validation"""
        current_price = self._get_current_price(technical)
        
        logger.info(f"🔍 SUPPORT CALC: Current price = {current_price}")
        logger.info(f"🔍 SUPPORT CALC: Technical data keys = {list(technical.keys())}")
        
        # Get basic indicators from nested structure
        basic_indicators = technical.get('basic_indicators', {})
        logger.info(f"🔍 SUPPORT CALC: basic_indicators = {basic_indicators}")
        
        # Try VWAP first
        vwap = basic_indicators.get('vwap', technical.get('vwap', technical.get('vwap_20')))
        logger.info(f"🔍 SUPPORT CALC: vwap = {vwap}")
        if vwap and vwap < current_price:
            logger.info(f"✅ SUPPORT CALC: Using vwap = {vwap}")
            return float(vwap)
        else:
            logger.info(f"❌ SUPPORT CALC: vwap rejected (value={vwap}, condition={vwap and vwap < current_price})")
        
        # Try 20-day SMA
        sma_20 = basic_indicators.get('sma_20', technical.get('sma_20', technical.get('sma20')))
        logger.info(f"🔍 SUPPORT CALC: sma_20 = {sma_20}")
        if sma_20 and sma_20 < current_price:
            logger.info(f"✅ SUPPORT CALC: Using sma_20 = {sma_20}")
            return float(sma_20)
        else:
            logger.info(f"❌ SUPPORT CALC: sma_20 rejected (value={sma_20}, condition={sma_20 and sma_20 < current_price})")
        
        # Try recent low
        recent_low = basic_indicators.get('recent_low', technical.get('recent_low', technical.get('low')))
        logger.info(f"🔍 SUPPORT CALC: recent_low = {recent_low}")
        if recent_low and recent_low < current_price and recent_low > 0:
            # Additional validation: reject obviously corrupted values
            # Support should be within reasonable range (not more than 10% below current price)
            if recent_low < current_price * 0.9:
                logger.warning(f"❌ SUPPORT CALC: Rejecting corrupted recent_low: {recent_low} (more than 10% below current price {current_price})")
            else:
                logger.info(f"✅ SUPPORT CALC: Using recent_low = {recent_low}")
                return float(recent_low)
        else:
            logger.info(f"❌ SUPPORT CALC: recent_low rejected (value={recent_low}, condition={recent_low and recent_low < current_price and recent_low > 0})")
        
        # Fallback to current price * 0.95 (5% downside assumption)
        support = current_price * 0.95
        logger.warning(f"⚠️ SUPPORT CALC: Using fallback calculation: {current_price} * 0.95 = {support}")
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
        elif ratio >= 1.5:
            risk_level = "Moderate Risk, Good Reward"
            advice = "Decent setup - risk ₹1 to make ₹1.50"
        elif ratio >= 1.0:
            risk_level = "Balanced Risk-Reward"
            advice = "Fair setup - risk ₹1 to make ₹1"
        else:
            risk_level = "High Risk, Low Reward"
            advice = "Poor setup - avoid or wait for better entry"
        
        # Get sample for context
        sample = list(real_money_impacts.keys())[0] if real_money_impacts else "₹10,000"
        sample_data = real_money_impacts.get(sample, {})
        
        return {
            "risk_level": risk_level,
            "advice": advice,
            "example": f"For {sample}: {sample_data.get('net_risk_reward', 'N/A')}",
            "ratio_meaning": f"{ratio:.2f}:1 ratio"
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
            
            return self._parse_json_response(response, "Top Drivers Analysis")
            
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
        
    def analyze_stock(self, symbol: str, fundamentals: Dict[str, Any], 
                     technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        OPTIMIZED: Main entry point for multi-stage analysis with single data processing
        Returns comprehensive analysis with all stages
        """
        try:
            logger.info(f"🔍 Multi-stage analysis for {symbol} using {self.ai_provider.upper()}")
            
            # OPTIMIZATION: Prepare all data once and reuse across stages
            processed_data = self._prepare_data_once(symbol, fundamentals, technical, enhanced_fundamentals)
            if not processed_data:
                return {"error": "Failed to prepare data for analysis"}
            
            logger.info(f"✅ Data processed once for {symbol} - reusing across all stages")
            
            # Stage 1: Simple Analysis (using pre-processed data)
            simple_analysis = self._stage1_simple_analysis(symbol, processed_data["fundamentals"], processed_data["technical"], processed_data["enhanced_fundamentals"], processed_data)
            if not simple_analysis:
                return {"error": "Stage 1 analysis failed"}
            
            # Stage 2: Simple Decision (using simple analysis output as input)
            simple_decision = self._stage2_simple_decision(symbol, simple_analysis, processed_data["fundamentals"], processed_data["technical"], processed_data["enhanced_fundamentals"], processed_data)
            if not simple_decision:
                return {"error": "Stage 2 analysis failed"}
            
            # NEW: Identify top drivers from all stored data
            top_drivers = self._identify_top_drivers(symbol, processed_data["technical"], processed_data["fundamentals"], processed_data["enhanced_fundamentals"])
            
            # Create compatibility layer for existing pipeline
            forensic_analysis = {
                "the_setup": simple_analysis.get("the_setup", ""),
                "the_catalyst": simple_analysis.get("the_catalyst", ""),
                "the_confirmation": simple_analysis.get("the_confirmation", ""),
                "overall_signal": simple_analysis.get("overall_signal", "neutral"),
                "confidence": simple_analysis.get("confidence", 0.0),
                "fundamental_score": simple_analysis.get("fundamental_score", 0.0),
                "technical_score": simple_analysis.get("technical_score", 0.0),
                "overall_signal_strength": simple_analysis.get("overall_signal_strength", 0.0),
                "primary_driver": simple_analysis.get("primary_driver", "momentum")
            }
            
            module_analysis = {
                "selected_module": "simple",
                "module_analysis": simple_analysis,
                "selection_reasoning": "Simplified 3-factor analysis"
            }
            
            risk_assessment = {
                "risk_level": simple_decision.get("risk_level", "moderate"),
                "deal_breakers": simple_decision.get("deal_breakers", []),
                "manageable_risks": simple_decision.get("manageable_risks", [])
            }
            
            final_decision = {
                "action": simple_decision.get("decision", "WATCH"),
                "confidence": simple_decision.get("confidence", 0.0),
                "rationale": simple_decision.get("reasoning", []),
                "position_size": simple_decision.get("position_size", "50%"),
                "stop_loss": simple_decision.get("stop_loss", 0.0),
                "target_price": simple_decision.get("target_price", 0.0)
            }
            
            # Compile comprehensive result
            result = {
                "symbol": symbol,
                "analysis_stages": {
                    "forensic_analysis": forensic_analysis,
                    "module_selection": module_analysis,
                    "risk_assessment": risk_assessment,
                    "final_decision": final_decision
                },
                "final_recommendation": {
                    "action": final_decision["action"],
                    "confidence": final_decision["confidence"],
                    "rationale": final_decision["rationale"],
                    "position_size": final_decision.get("position_size", "50%"),
                    "stop_loss": final_decision.get("stop_loss"),
                    "target_price": final_decision.get("target_price"),
                    "holding_period": "medium-term"
                },
                "risk_summary": {
                    "risk_level": risk_assessment["risk_level"],
                    "deal_breakers": risk_assessment.get("deal_breakers", []),
                    "manageable_risks": risk_assessment.get("manageable_risks", [])
                },
                "simple_analysis": simple_analysis,
                "simple_decision": simple_decision,
                # NEW: Top drivers analysis
                "top_drivers": top_drivers
            }
            
            logger.info(f"✅ Multi-stage analysis completed for {symbol}: {final_decision['action']} (confidence: {final_decision['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-stage analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def _stage1_simple_analysis(self, symbol: str, fundamentals: Dict[str, Any], 
                               technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None, processed_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        ENHANCED: Stage 1: Simple 3-Factor Analysis using pre-processed data
        Identifies the 3 most important factors: Setup, Catalyst, Confirmation
        """
        try:
            # OPTIMIZATION: Use pre-processed data if available
            if processed_data:
                logger.debug(f"🎯 Using pre-processed data for Stage 1 {symbol}")
                anonymized_data = processed_data["anonymized_data"]
                formatted_data = processed_data["formatted_data"]
                risk_reward = processed_data["risk_reward"]
            else:
                # Fallback to original processing
                logger.debug(f"📊 Processing data for Stage 1 {symbol}")
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

Focus on the 3-factor analysis framework only.

Return JSON(Example):
{{
    "the_setup": "text",
    "the_catalyst": "text", 
    "the_confirmation": "text",
    "overall_signal": "strong/weak/neutral",
    "confidence": 0.0,
    "fundamental_score": 0.0,
    "technical_score": 0.0,
    "overall_signal_strength": 0.0,
    "primary_driver": "momentum"
}}"""
            
            response = self._call_ai_provider(prompt, "You are a stock analyst who identifies the most important factors in simple terms.")
            if not response:
                return None
            result = self._parse_json_response(response, "Stage 1 Simple Analysis")
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
            
            # COST-OPTIMIZED PROMPT - Much shorter
            prompt = f"""DECISION: {symbol}

Setup: {the_setup}
Catalyst: {the_catalyst}  
Confirmation: {the_confirmation}
Signal: {overall_signal} (confidence: {confidence})

Make clear decision: BUY/WATCH/AVOID

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
            
            response = self._call_ai_provider(prompt, "You are a decision maker who makes clear buy/watch/avoid decisions.")
            if not response:
                return None
                
            return self._parse_json_response(response, "Stage 2 Simple Decision")
            
        except Exception as e:
            logger.error(f"Stage 2 simple decision failed for {symbol}: {e}")
            return None
    
    def _momentum_module_analysis(self, symbol: str, forensic_analysis: Dict[str, Any], 
                                 fundamentals: Dict[str, Any], technical: Dict[str, Any], 
                                 enhanced_fundamentals: Dict[str, Any] = None, processed_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """OPTIMIZED: Momentum Module using pre-processed data"""
        # OPTIMIZATION: Use pre-processed data if available
        if processed_data:
            logger.debug(f"🎯 Using pre-processed data for Momentum Module {symbol}")
            anonymized_data = processed_data["anonymized_data"]
            formatted_data = processed_data["formatted_data"]
            risk_reward = processed_data["risk_reward"]
        else:
            # Fallback to original processing
            logger.debug(f"📊 Processing data for Momentum Module {symbol}")
            # anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
            # Use raw data instead of anonymized data
            anonymized_data = {
                "fundamentals": fundamentals,
                "technical": technical,
                "enhanced_fundamentals": enhanced_fundamentals or {},
                "symbol": symbol
            }
            formatted_data = self._format_data_for_analysis(anonymized_data)
            risk_reward = self._calculate_risk_reward_once(symbol, technical)
        
        prompt = f"""
MOMENTUM MODULE ANALYSIS - Is this strong momentum or a bull trap?

=== DATA ===
{formatted_data}

=== FORENSIC ANALYSIS CONTEXT ===
{json.dumps(forensic_analysis, indent=2)}

=== MOMENTUM MODULE FOCUS ===

OPTIMIZED: RISK-REWARD CALCULATION (PRE-CALCULATED)
```
Given:
- Current Price: ₹{risk_reward.get('current_price', 'N/A')}
- Resistance Level: ₹{risk_reward.get('resistance_level', 'N/A')}
- Support Level: ₹{risk_reward.get('support_level', 'N/A')}

Pre-calculated Results:
- Upside: ₹{risk_reward.get('upside', 'N/A')} ({risk_reward.get('upside_percentage', 'N/A')}%)
- Downside: ₹{risk_reward.get('downside', 'N/A')} ({risk_reward.get('downside_percentage', 'N/A')}%)
- Risk-Reward Ratio: {risk_reward.get('risk_reward_ratio', 'N/A')}:1
- Interpretation: {risk_reward.get('ratio_interpretation', 'N/A')}
- Calculation Steps: {risk_reward.get('calculation_steps', 'N/A')}

OPTIMIZATION: Risk-reward calculation is pre-computed and provided above.
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
        # anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
        # Use raw data instead of anonymized data
        anonymized_data = {
            "fundamentals": fundamentals,
            "technical": technical,
            "enhanced_fundamentals": enhanced_fundamentals or {},
            "symbol": symbol
        }
        
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
        # anonymized_data = self.openai_client._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
        # Use raw data instead of anonymized data
        anonymized_data = {
            "fundamentals": fundamentals,
            "technical": technical,
            "enhanced_fundamentals": enhanced_fundamentals or {},
            "symbol": symbol
        }
        
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
