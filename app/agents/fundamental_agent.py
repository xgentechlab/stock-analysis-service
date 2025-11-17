"""
Fundamental Agent - Analyzes company fundamentals
"""
from typing import Dict, Any, Optional
import logging

from app.agents.base_agent import BaseAgent
from app.agents.models import Verdict, AgentResult
from app.services.stocks import stocks_service
from app.services.enhanced_fundamentals import enhanced_fundamental_analysis
from app.services.fundamental_scoring import fundamental_scoring

logger = logging.getLogger(__name__)


class FundamentalAgent(BaseAgent):
    """Agent for fundamental analysis"""
    
    def __init__(self):
        super().__init__("fundamental", ttl_hours=168)  # 7 days
    
    def fetch_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data"""
        try:
            ticker_data = stocks_service._fetch_all_data_optimized(symbol)
            if not ticker_data:
                return {}
            
            enhanced_fundamentals = self._process_fundamentals(
                symbol, ticker_data
            )
            score_data = self._calculate_fundamental_score(
                enhanced_fundamentals
            )
            
            return self._build_result_dict(
                enhanced_fundamentals, score_data, ticker_data
            )
        except Exception as e:
            logger.error(f"Error fetching fundamental data: {e}")
            return {}
    
    def _calculate_fundamental_score(self, fundamentals: Dict) -> Dict:
        """Calculate fundamental score"""
        return fundamental_scoring.calculate_fundamental_score(
            fundamentals
        )
    
    def _build_result_dict(
        self, fundamentals: Dict, score_data: Dict, ticker_data: Dict
    ) -> Dict[str, Any]:
        """Build result dictionary"""
        return {
            'metrics': fundamentals,
            'analysis': score_data,
            'basic_fundamentals': ticker_data.get("basic_fundamentals", {})
        }
    
    def _process_fundamentals(
        self, symbol: str, ticker_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process fundamental data from ticker"""
        info = ticker_data.get("info", {})
        financials = ticker_data.get("financials")
        balance_sheet = ticker_data.get("balance_sheet")
        cash_flow = ticker_data.get("cash_flow")
        
        return enhanced_fundamental_analysis._process_with_prefetched_data(
            symbol, info, financials, balance_sheet, cash_flow
        )
    
    def calculate_score(self, data: Dict[str, Any]) -> float:
        """Calculate fundamental score"""
        analysis = data.get('analysis', {})
        return float(analysis.get('final_score', 0.5))
    
    def generate_verdict(self, score: float) -> Verdict:
        """Generate verdict from score"""
        if score >= 0.7:
            return Verdict.BUY
        elif score <= 0.3:
            return Verdict.SELL
        else:
            return Verdict.NEUTRAL
    
    def calculate_confidence(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> float:
        """Calculate confidence with AI enhancement"""
        analysis = data.get('analysis', {})
        base_confidence = float(analysis.get('confidence', 0.5))
        
        metrics = data.get('metrics', {})
        data_completeness = self._check_data_completeness(metrics)
        
        base_conf = min(1.0, base_confidence * data_completeness)
        
        # AI-enhanced confidence adjustment
        ai_confidence_boost = self._get_ai_confidence_adjustment(
            data, score, base_conf
        )
        
        return min(1.0, base_conf + ai_confidence_boost)
    
    def _get_ai_confidence_adjustment(
        self, data: Dict[str, Any], score: float, base_confidence: float
    ) -> float:
        """Get AI-based confidence adjustment"""
        try:
            # Use data quality and signal strength for AI confidence
            metrics = data.get('metrics', {})
            data_completeness = self._check_data_completeness(metrics)
            
            # Signal strength: how far from neutral (0.5)
            signal_strength = abs(score - 0.5) * 2  # 0-1 scale
            
            # AI confidence boost: higher for clear signals with good data
            boost = (signal_strength * data_completeness) * 0.1
            
            return boost
        except Exception as e:
            logger.warning(f"AI confidence adjustment failed: {e}")
            return 0.0
    
    def extract_key_factors(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> list:
        """Extract key factors"""
        factors = []
        metrics = data.get('metrics', {})
        
        factors.extend(self._extract_quality_factors(metrics))
        factors.extend(self._extract_value_factors(metrics))
        factors.extend(self._extract_growth_factors(metrics))
        
        return factors[:5]
    
    def _extract_quality_factors(self, metrics: Dict) -> list:
        """Extract quality factors"""
        factors = []
        quality = metrics.get('quality_metrics', {})
        if quality.get('roe_consistency', 0) > 15:
            factors.append("Strong ROE consistency")
        return factors
    
    def _extract_value_factors(self, metrics: Dict) -> list:
        """Extract value factors"""
        factors = []
        value = metrics.get('value_metrics', {})
        pe = value.get('pe_ratio')
        if pe and pe < 20:
            factors.append(f"Attractive P/E ratio ({pe:.1f})")
        return factors
    
    def _extract_growth_factors(self, metrics: Dict) -> list:
        """Extract growth factors"""
        factors = []
        growth = metrics.get('growth_metrics', {})
        if growth.get('revenue_growth_yoy', 0) > 10:
            factors.append("Strong revenue growth")
        return factors
    
    def _check_data_completeness(self, metrics: Dict) -> float:
        """Check data completeness for confidence"""
        required = ['quality_metrics', 'value_metrics', 'growth_metrics']
        present = sum(1 for key in required if key in metrics)
        return present / len(required)
    
    def analyze(self, symbol: str) -> AgentResult:
        """Override analyze to add AI explanation"""
        try:
            data = self.fetch_data(symbol)
            if not data:
                raise ValueError(f"No data fetched for {symbol}")
            
            score = self.calculate_score(data)
            verdict = self.generate_verdict(score)
            confidence = self.calculate_confidence(data, score)
            key_factors = self.extract_key_factors(data, score)
            
            # Generate AI explanation
            ai_explanation = self._generate_ai_explanation(
                symbol, data, score, verdict
            )
            
            return self._create_result(
                symbol, score, verdict, confidence, key_factors, data, ai_explanation
            )
        except Exception as e:
            logger.error(f"{self.name} failed for {symbol}: {e}")
            raise
    
    def _generate_ai_explanation(
        self, symbol: str, data: Dict[str, Any],
        score: float, verdict: Verdict
    ) -> Optional[str]:
        """Generate AI-powered explanation"""
        try:
            from app.agents.agent_ai_reasoning import agent_ai_reasoning
            metrics = data.get('metrics', {})
            return agent_ai_reasoning.generate_fundamental_explanation(
                symbol, metrics, score, verdict.value
            )
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            return None

