"""
Data models for agent system
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    """Agent verdict types"""
    BUY = "BUY"
    SELL = "SELL"
    WATCH = "WATCH"
    NEUTRAL = "NEUTRAL"


@dataclass
class AgentResult:
    """Result from an agent analysis"""
    agent_name: str
    symbol: str
    score: float
    verdict: Verdict
    confidence: float
    key_factors: List[str]
    metrics: Dict[str, Any]
    analysis: Dict[str, Any]
    ai_explanation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'agent_name': self.agent_name,
            'symbol': self.symbol,
            'score': self.score,
            'verdict': self.verdict.value,
            'confidence': self.confidence,
            'key_factors': self.key_factors,
            'metrics': self.metrics,
            'analysis': self.analysis,
            'ai_explanation': self.ai_explanation,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class FinalRecommendation:
    """Final recommendation combining all agents"""
    symbol: str
    recommendation: Verdict
    confidence: float
    reasoning: str
    agent_breakdown: Dict[str, AgentResult]
    weighted_votes: Dict[str, float]
    trade_details: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'recommendation': self.recommendation.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'agent_breakdown': {
                name: result.to_dict() 
                for name, result in self.agent_breakdown.items()
            },
            'weighted_votes': self.weighted_votes,
            'trade_details': self.trade_details,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

