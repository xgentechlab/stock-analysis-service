"""
AI Reasoning for Individual Agents
Generates AI-powered explanations for Fundamental and Technical agents
"""
from typing import Dict, Any, Optional
import logging
import json

from app.config import settings
from app.services.openai_client import openai_client
from app.services.claude_client import claude_client

logger = logging.getLogger(__name__)


class AgentAIReasoning:
    """AI reasoning for individual agents"""
    
    def __init__(self):
        self.provider = settings.ai_provider.lower()
        self.use_openai = self.provider == "openai"
        self.use_claude = self.provider == "claude"
    
    def generate_fundamental_explanation(
        self, symbol: str, metrics: Dict[str, Any],
        score: float, verdict: str
    ) -> str:
        """Generate AI explanation for fundamental analysis"""
        try:
            prompt = self._build_fundamental_prompt(
                symbol, metrics, score, verdict
            )
            
            if self.use_openai:
                return self._call_openai(prompt, "fundamental")
            elif self.use_claude:
                return self._call_claude(prompt, "fundamental")
        except Exception as e:
            logger.error(f"AI fundamental explanation failed: {e}")
        
        return self._fallback_fundamental_explanation(metrics, score)
    
    def generate_technical_explanation(
        self, symbol: str, metrics: Dict[str, Any],
        score: float, verdict: str
    ) -> str:
        """Generate AI explanation for technical analysis"""
        try:
            prompt = self._build_technical_prompt(
                symbol, metrics, score, verdict
            )
            
            if self.use_openai:
                return self._call_openai(prompt, "technical")
            elif self.use_claude:
                return self._call_claude(prompt, "technical")
        except Exception as e:
            logger.error(f"AI technical explanation failed: {e}")
        
        return self._fallback_technical_explanation(metrics, score)
    
    def _build_fundamental_prompt(
        self, symbol: str, metrics: Dict[str, Any],
        score: float, verdict: str
    ) -> str:
        """Build prompt for fundamental analysis"""
        quality = metrics.get('quality_metrics', {})
        value = metrics.get('value_metrics', {})
        growth = metrics.get('growth_metrics', {})
        
        return f"""Analyze the fundamental metrics for {symbol} and provide a clear explanation.

VERDICT: {verdict} (Score: {score:.2f})

QUALITY METRICS:
- ROE: {quality.get('roe_consistency', 'N/A')}%
- Debt/Equity: {quality.get('debt_equity_ratio', 'N/A')}
- Operating Margin: {quality.get('operating_margin', 'N/A')}%
- Current Ratio: {quality.get('current_ratio', 'N/A')}

VALUE METRICS:
- P/E Ratio: {value.get('pe_ratio', 'N/A')}
- P/B Ratio: {value.get('pb_ratio', 'N/A')}
- Dividend Yield: {value.get('dividend_yield', 'N/A')}%

GROWTH METRICS:
- Revenue CAGR: {growth.get('revenue_cagr', 'N/A')}%
- EPS Growth: {growth.get('eps_growth_yoy', 'N/A')}%

Provide a 2-3 sentence explanation in plain English:
1. What the fundamental health looks like
2. Key strengths or weaknesses
3. Why this leads to a {verdict} recommendation

Use simple language, avoid jargon."""
    
    def _build_technical_prompt(
        self, symbol: str, metrics: Dict[str, Any],
        score: float, verdict: str
    ) -> str:
        """Build prompt for technical analysis"""
        current_price = metrics.get('close', metrics.get('current_price', 'N/A'))
        rsi = metrics.get('rsi_14', metrics.get('rsi14', 'N/A'))
        macd = metrics.get('macd', 'N/A')
        sma_20 = metrics.get('sma_20', metrics.get('sma20', 'N/A'))
        support = metrics.get('support_level', 'N/A')
        resistance = metrics.get('resistance_level', 'N/A')
        
        return f"""Analyze the technical indicators for {symbol} and provide a clear explanation.

VERDICT: {verdict} (Score: {score:.2f})

PRICE & LEVELS:
- Current Price: ₹{current_price}
- Support Level: ₹{support}
- Resistance Level: ₹{resistance}

MOMENTUM:
- RSI (14): {rsi}
- MACD: {macd}
- Price vs SMA 20: {((current_price - sma_20) / sma_20 * 100) if isinstance(current_price, (int, float)) and isinstance(sma_20, (int, float)) and sma_20 > 0 else 'N/A'}%

Provide a 2-3 sentence explanation in plain English:
1. What the price action and momentum suggest
2. Key technical patterns or signals
3. Why this leads to a {verdict} recommendation

Use simple language, explain what the indicators mean for investors."""
    
    def _call_openai(self, prompt: str, agent_type: str) -> str:
        """Call OpenAI API"""
        if not openai_client.client:
            raise ValueError("OpenAI client not initialized")
        
        system_msg = (
            "You are a financial analyst explaining stock analysis in simple terms."
            if agent_type == "fundamental"
            else "You are a technical analyst explaining price patterns in simple terms."
        )
        
        response = openai_client.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()[:300]
    
    def _call_claude(self, prompt: str, agent_type: str) -> str:
        """Call Claude API"""
        if not claude_client.client:
            raise ValueError("Claude client not initialized")
        
        system_msg = (
            "You are a financial analyst explaining stock analysis in simple terms."
            if agent_type == "fundamental"
            else "You are a technical analyst explaining price patterns in simple terms."
        )
        
        response = claude_client.client.messages.create(
            model=settings.claude_model,
            max_tokens=200,
            temperature=0.3,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()[:300]
    
    def _fallback_fundamental_explanation(
        self, metrics: Dict[str, Any], score: float
    ) -> str:
        """Fallback explanation for fundamentals"""
        if score >= 0.7:
            return "Strong fundamental metrics indicate good financial health."
        elif score <= 0.3:
            return "Weak fundamental metrics suggest financial concerns."
        return "Fundamental metrics are mixed, showing neutral financial health."
    
    def _fallback_technical_explanation(
        self, metrics: Dict[str, Any], score: float
    ) -> str:
        """Fallback explanation for technicals"""
        if score >= 0.65:
            return "Technical indicators show bullish momentum."
        elif score <= 0.35:
            return "Technical indicators show bearish signals."
        return "Technical indicators are mixed, showing neutral momentum."


# Singleton instance
agent_ai_reasoning = AgentAIReasoning()

