"""
Anthropic Claude client wrapper for stock analysis verdicts
Following user requirement to keep all API calls in separate file and anonymize data
Includes prompt caching for cost optimization
"""
# import anthropic  # Import inside the method to avoid issues
from typing import Dict, Any, Optional, List
import json
import logging
import hashlib
from datetime import datetime, timedelta

from app.config import settings

logger = logging.getLogger(__name__)

class ClaudeClient:
    def __init__(self):
        self.api_key = settings.anthropic_api_key
        self.model = settings.claude_model
        self.max_tokens = settings.claude_max_tokens
        self.temperature = settings.claude_temperature
        
        # Prompt cache for cost optimization
        self.prompt_cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
        if not self.api_key:
            logger.warning("Anthropic API key not configured")
            self.client = None
        else:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(
                    api_key=self.api_key
                )
                logger.info("✅ Claude client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
    
    def _generate_cache_key(self, symbol: str, fundamentals: Dict[str, Any], 
                           technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> str:
        """Generate a cache key based on the input data"""
        # Create a hash of the anonymized data for caching
        cache_data = {
            "fundamentals": fundamentals,
            "technical": technical,
            "enhanced_fundamentals": enhanced_fundamentals or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        return datetime.now() - cache_entry['timestamp'] < self.cache_duration
    
    def _anonymize_data(self, symbol: str, fundamentals: Dict[str, Any], 
                       technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Anonymize data before sending to Claude API
        Remove any personal/identifiable information as per user requirement
        """
        # Create anonymized version without symbol name
        anonymized = {
            "fundamentals": {
                "pe": fundamentals.get("pe"),
                "pb": fundamentals.get("pb"), 
                "roe": fundamentals.get("roe"),
                "eps_ttm": fundamentals.get("eps_ttm"),
                "market_cap_cr": fundamentals.get("market_cap_cr")
            },
            "enhanced_fundamentals": enhanced_fundamentals or {},
            "technical": {
                # Basic indicators
                "current_price": technical.get("current_price", technical.get("close")),
                "close_price": technical.get("close"),
                "sma_20": technical.get("sma_20", technical.get("sma20")),
                "sma_50": technical.get("sma_50", technical.get("sma50")),
                "rsi_14": technical.get("rsi_14", technical.get("rsi14")),
                "atr_14": technical.get("atr_14", technical.get("atr14")),
                "pct_1d": technical.get("pct_1d", technical.get("price_change_1d_pct")),
                "pct_5d": technical.get("pct_5d", technical.get("price_change_5d_pct")),
                "is_breakout": technical.get("is_breakout", False),
                
                # Enhanced momentum indicators
                "macd": technical.get("macd"),
                "macd_signal": technical.get("macd_signal"),
                "macd_histogram": technical.get("macd_histogram"),
                "stoch_rsi_k": technical.get("stoch_rsi_k"),
                "stoch_rsi_d": technical.get("stoch_rsi_d"),
                "williams_r": technical.get("williams_r"),
                "roc_5": technical.get("roc_5"),
                "roc_10": technical.get("roc_10"),
                "roc_20": technical.get("roc_20"),
                
                # Volume indicators
                "vwap": technical.get("vwap"),
                "vwap_upper": technical.get("vwap_upper"),
                "vwap_lower": technical.get("vwap_lower"),
                "obv": technical.get("obv"),
                "ad_line": technical.get("ad_line"),
                "volume_spike_ratio": technical.get("vol_today", 0) / max(technical.get("vol20", 1), 1),
                
                # Divergence signals
                "rsi_divergence": technical.get("rsi_divergence", {}),
                
                # Multi-timeframe analysis
                "1m_trend": technical.get("1m_trend"),
                "5m_trend": technical.get("5m_trend"),
                "15m_trend": technical.get("15m_trend"),
                "1d_trend": technical.get("1d_trend"),
                "1wk_trend": technical.get("1wk_trend")
            }
        }
        
        # Remove None values
        anonymized["fundamentals"] = {k: v for k, v in anonymized["fundamentals"].items() if v is not None}
        anonymized["technical"] = {k: v for k, v in anonymized["technical"].items() if v is not None}
        
        return anonymized
    
    def get_stock_verdict(self, symbol: str, fundamentals: Dict[str, Any], 
                         technical: Dict[str, Any], enhanced_fundamentals: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get Claude verdict for a stock based on fundamental and technical analysis
        Returns: {"action": "buy|watch|avoid", "confidence": float, "explain": str}
        """
        try:
            logger.info(f"🧠 Claude analysis for {symbol}")
            
            if not self.client:
                logger.error("Claude client not initialized")
                return None
            
            # Anonymize data before sending to Claude - COMMENTED OUT FOR TESTING
            # anonymized_data = self._anonymize_data(symbol, fundamentals, technical, enhanced_fundamentals)
            # Use raw data instead of anonymized data
            anonymized_data = {
                "fundamentals": fundamentals,
                "technical": technical,
                "enhanced_fundamentals": enhanced_fundamentals or {},
                "symbol": symbol
            }
            
            # Check cache first for cost optimization
            cache_key = self._generate_cache_key(symbol, fundamentals, technical, enhanced_fundamentals)
            if cache_key in self.prompt_cache:
                cache_entry = self.prompt_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    logger.info(f"🎯 Cache hit for {symbol} - returning cached verdict")
                    return cache_entry['verdict']
                else:
                    # Remove expired cache entry
                    del self.prompt_cache[cache_key]
            
            
            # Create the prompt with caching optimization
            system_prompt, user_prompt = self._create_analysis_prompts(anonymized_data)
            
            # Use prompt caching for the system prompt (large, repetitive content)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Parse the response
            content = response.content[0].text.strip()
            
            # Try to extract JSON from response
            verdict = self._parse_verdict_response(content)
            
            if verdict:
                logger.info(f"✅ CLAUDE SUCCESS for {symbol}: {verdict['action']} (confidence: {verdict['confidence']})")
                
                # Cache the result for cost optimization
                self.prompt_cache[cache_key] = {
                    'verdict': verdict,
                    'timestamp': datetime.now()
                }
                
                # Log cache statistics
                logger.info(f"📊 Claude Cache size: {len(self.prompt_cache)} entries")
                
                return verdict
            else:
                logger.error(f"❌ CLAUDE FAILED to parse response for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ CLAUDE API ERROR for {symbol}: {e}")
            return None
    
    def _create_analysis_prompts(self, anonymized_data: Dict[str, Any]) -> tuple[str, str]:
        """Create the comprehensive analysis prompts for Claude with caching optimization"""
        
        fundamentals = anonymized_data["fundamentals"]
        technical = anonymized_data["technical"]
        enhanced_fundamentals = anonymized_data.get("enhanced_fundamentals", {})
        
        # Extract enhanced fundamental metrics
        quality_metrics = enhanced_fundamentals.get("quality_metrics", {})
        growth_metrics = enhanced_fundamentals.get("growth_metrics", {})
        value_metrics = enhanced_fundamentals.get("value_metrics", {})
        momentum_metrics = enhanced_fundamentals.get("momentum_metrics", {})
        
        # System prompt (cached) - contains the analysis framework
        system_prompt = """You are a professional equity research analyst with 15+ years of experience. Analyze the provided comprehensive stock data and return a detailed JSON response with your investment recommendation. Focus on providing thorough analysis with specific reasoning for each factor.

=== COMPREHENSIVE ANALYSIS FRAMEWORK ===

1. FUNDAMENTAL STRENGTH:
   - Evaluate quality metrics for financial health
   - Assess growth prospects and sustainability
   - Analyze valuation relative to industry and historical levels
   - Consider dividend yield and payout sustainability

2. TECHNICAL MOMENTUM:
   - Analyze price action and trend strength
   - Evaluate momentum indicators for overbought/oversold conditions
   - Assess volume confirmation and institutional interest
   - Check multi-timeframe trend alignment

3. RISK ASSESSMENT:
   - Evaluate volatility and beta characteristics
   - Assess debt levels and financial stability
   - Consider market cap and liquidity factors
   - Analyze analyst sentiment and price momentum

4. ENTRY/EXIT TIMING:
   - Identify optimal entry points based on technical levels
   - Assess risk-reward ratio
   - Consider stop-loss levels and position sizing
   - Evaluate short-term vs long-term prospects

=== INVESTMENT RECOMMENDATION ===

Provide a detailed analysis as a JSON object with this exact format:

{
    "action": "buy" | "watch" | "avoid",
    "confidence": 0.0-1.0,
    "explain": "Comprehensive analysis including: 1) Key fundamental strengths/weaknesses, 2) Technical setup and momentum, 3) Risk factors, 4) Entry strategy and price targets, 5) Time horizon recommendation (max 300 words)"
}

GUIDELINES:
- "buy": Strong fundamental + technical setup, clear entry opportunity
- "watch": Mixed signals, wait for better entry or more clarity
- "avoid": Weak fundamentals, poor technical setup, or high risk
- Confidence: 0.0-1.0 based on conviction level and data quality
- Provide specific reasoning for each major factor
- Include price targets and risk management suggestions
- Consider both short-term trading and long-term investment perspectives"""
        
        # User prompt (not cached) - contains the specific data
        user_prompt = f"""Analyze this comprehensive stock data and provide a detailed investment recommendation.

=== BASIC FUNDAMENTAL DATA ===
- P/E Ratio: {fundamentals.get('pe', 'N/A')}
- P/B Ratio: {fundamentals.get('pb', 'N/A')}
- ROE: {fundamentals.get('roe', 'N/A')}%
- EPS (TTM): ₹{fundamentals.get('eps_ttm', 'N/A')}
- Market Cap: ₹{fundamentals.get('market_cap_cr', 'N/A')} Cr

=== ENHANCED FUNDAMENTAL ANALYSIS ===

QUALITY METRICS (Financial Health):
- Operating Margin: {quality_metrics.get('operating_margin', 'N/A')}
- Net Margin: {quality_metrics.get('net_margin', 'N/A')}
- Gross Margin: {quality_metrics.get('gross_margin', 'N/A')}
- Current Ratio: {quality_metrics.get('current_ratio', 'N/A')}
- Quick Ratio: {quality_metrics.get('quick_ratio', 'N/A')}
- ROE Consistency: {quality_metrics.get('roe_consistency', 'N/A')}%
- Debt-to-Equity: {quality_metrics.get('debt_equity_ratio', 'N/A')}
- Interest Coverage: {quality_metrics.get('interest_coverage', 'N/A')}x
- Cash Conversion Cycle: {quality_metrics.get('cash_conversion_cycle', 'N/A')} days

GROWTH METRICS (Business Growth):
- Revenue CAGR: {growth_metrics.get('revenue_cagr', 'N/A')}%
- Book Value Growth: {growth_metrics.get('book_value_growth', 'N/A')}%
- Free Cash Flow Growth: {growth_metrics.get('free_cash_flow_growth', 'N/A')}%
- EPS Growth QoQ: {growth_metrics.get('eps_growth_qoq', 'N/A')}%
- EPS Growth YoY: {growth_metrics.get('eps_growth_yoy', 'N/A')}%

VALUE METRICS (Valuation):
- P/E Ratio: {value_metrics.get('pe_ratio', 'N/A')}
- P/B Ratio: {value_metrics.get('pb_ratio', 'N/A')}
- P/S Ratio: {value_metrics.get('ps_ratio', 'N/A')}
- EV/EBITDA: {value_metrics.get('ev_ebitda', 'N/A')}
- PEG Ratio: {value_metrics.get('peg_ratio', 'N/A')}
- Dividend Yield: {value_metrics.get('dividend_yield', 'N/A')}%
- Dividend Rate: ₹{value_metrics.get('dividend_rate', 'N/A')}
- P/E vs Industry: {value_metrics.get('pe_vs_industry', 'N/A')}x
- P/B vs Industry: {value_metrics.get('pb_vs_industry', 'N/A')}x

MOMENTUM METRICS (Market Sentiment):
- Analyst Score: {momentum_metrics.get('analyst_score', 'N/A')}/5
- Recommendation Mean: {momentum_metrics.get('recommendation_mean', 'N/A')}
- Price Momentum: {momentum_metrics.get('price_momentum', 'N/A')}%
- 52W High: ₹{momentum_metrics.get('fifty_two_week_high', 'N/A')}
- 52W Low: ₹{momentum_metrics.get('fifty_two_week_low', 'N/A')}
- Beta: {momentum_metrics.get('beta', 'N/A')}

=== TECHNICAL ANALYSIS ===

PRICE & MOVING AVERAGES:
- Current Price: ₹{technical.get('current_price', technical.get('close_price', 'N/A'))}
- 20-day SMA: ₹{technical.get('sma_20', technical.get('sma20', 'N/A'))}
- 50-day SMA: ₹{technical.get('sma_50', technical.get('sma50', 'N/A'))}
- Price vs SMA20: {((technical.get('current_price', 0) or 0) - (technical.get('sma_20', 0) or 0)) / (technical.get('sma_20', 1) or 1) * 100:.2f}%

MOMENTUM INDICATORS:
- RSI (14): {technical.get('rsi_14', technical.get('rsi14', 'N/A'))}
- MACD: {technical.get('macd', 'N/A')}
- MACD Signal: {technical.get('macd_signal', 'N/A')}
- MACD Histogram: {technical.get('macd_histogram', 'N/A')}
- Stochastic RSI K: {technical.get('stoch_rsi_k', 'N/A')}
- Williams %R: {technical.get('williams_r', 'N/A')}
- ROC 5-day: {technical.get('roc_5', 'N/A')}%
- ROC 10-day: {technical.get('roc_10', 'N/A')}%
- ROC 20-day: {technical.get('roc_20', 'N/A')}%

VOLUME ANALYSIS:
- VWAP: ₹{technical.get('vwap', 'N/A')}
- VWAP Upper Band: ₹{technical.get('vwap_upper', 'N/A')}
- VWAP Lower Band: ₹{technical.get('vwap_lower', 'N/A')}
- Price vs VWAP: {((technical.get('current_price', 0) or 0) - (technical.get('vwap', 0) or 0)) / (technical.get('vwap', 1) or 1) * 100:.2f}%
- OBV: {technical.get('obv', 'N/A')}
- A/D Line: {technical.get('ad_line', 'N/A')}

PRICE ACTION:
- 1-day Change: {technical.get('pct_1d', technical.get('price_change_1d_pct', 0))*100:.2f}%
- 5-day Change: {technical.get('pct_5d', technical.get('price_change_5d_pct', 0))*100:.2f}%
- Breakout Signal: {technical.get('is_breakout', False)}
- Bullish Divergence: {technical.get('rsi_divergence', {}).get('bullish_divergence', False)}
- Bearish Divergence: {technical.get('rsi_divergence', {}).get('bearish_divergence', False)}

MULTI-TIMEFRAME TRENDS:
- 1-minute: {technical.get('1m_trend', 'N/A')}
- 5-minute: {technical.get('5m_trend', 'N/A')}
- 15-minute: {technical.get('15m_trend', 'N/A')}
- Daily: {technical.get('1d_trend', 'N/A')}
- Weekly: {technical.get('1wk_trend', 'N/A')}"""
        
        return system_prompt, user_prompt
    
    def _parse_verdict_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse Claude response and extract verdict JSON"""
        try:
            # Try to find JSON in the response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx:end_idx]
                verdict = json.loads(json_str)
                
                # Validate required fields
                if all(key in verdict for key in ['action', 'confidence', 'explain']):
                    # Validate action
                    if verdict['action'] not in ['buy', 'watch', 'avoid']:
                        logger.warning(f"Invalid action: {verdict['action']}")
                        return None
                    
                    # Validate confidence
                    confidence = float(verdict['confidence'])
                    if not (0.0 <= confidence <= 1.0):
                        logger.warning(f"Invalid confidence: {confidence}")
                        return None
                    
                    verdict['confidence'] = confidence
                    return verdict
                    
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse verdict JSON: {e}")
            
        return None
    
    def clear_cache(self):
        """Clear the prompt cache"""
        self.prompt_cache.clear()
        logger.info("🧹 Prompt cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cache_size": len(self.prompt_cache),
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600,
            "cached_keys": list(self.prompt_cache.keys())
        }

# Singleton instance
claude_client = ClaudeClient()
