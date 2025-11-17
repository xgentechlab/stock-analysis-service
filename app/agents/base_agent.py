"""
Base agent class for all analysis agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.agents.models import AgentResult, Verdict

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all analysis agents"""
    
    def __init__(self, name: str, ttl_hours: int):
        self.name = name
        self.ttl_hours = ttl_hours
    
    @abstractmethod
    def fetch_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch data required for analysis"""
        pass
    
    @abstractmethod
    def calculate_score(self, data: Dict[str, Any]) -> float:
        """Calculate score from data (0-1)"""
        pass
    
    @abstractmethod
    def generate_verdict(self, score: float) -> Verdict:
        """Generate verdict from score"""
        pass
    
    @abstractmethod
    def calculate_confidence(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> float:
        """Calculate confidence (0-1)"""
        pass
    
    @abstractmethod
    def extract_key_factors(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> list:
        """Extract key factors driving the score"""
        pass
    
    def analyze(self, symbol: str) -> AgentResult:
        """Main analysis method"""
        try:
            data = self.fetch_data(symbol)
            if not data:
                raise ValueError(f"No data fetched for {symbol}")
            
            score = self.calculate_score(data)
            verdict = self.generate_verdict(score)
            confidence = self.calculate_confidence(data, score)
            key_factors = self.extract_key_factors(data, score)
            
            return self._create_result(
                symbol, score, verdict, confidence, key_factors, data
            )
        except Exception as e:
            logger.error(f"{self.name} failed for {symbol}: {e}")
            raise
    
    def _create_result(
        self, symbol: str, score: float, verdict: Verdict,
        confidence: float, key_factors: list, data: Dict[str, Any],
        ai_explanation: Optional[str] = None
    ) -> AgentResult:
        """Create agent result object"""
        expires_at = datetime.utcnow() + timedelta(hours=self.ttl_hours)
        
        return AgentResult(
            agent_name=self.name,
            symbol=symbol,
            score=score,
            verdict=verdict,
            confidence=confidence,
            key_factors=key_factors,
            metrics=self._extract_metrics(data),
            analysis=self._extract_analysis(data, score),
            ai_explanation=ai_explanation,
            expires_at=expires_at
        )
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from data"""
        return data.get('metrics', {})
    
    def _extract_analysis(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> Dict[str, Any]:
        """Extract analysis details"""
        return data.get('analysis', {})

