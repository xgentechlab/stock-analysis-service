"""
Market Agent - Orchestrates all agents and provides market context
"""
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.base_agent import BaseAgent
from app.agents.models import AgentResult, FinalRecommendation, Verdict
from app.agents.fundamental_agent import FundamentalAgent
from app.agents.technical_agent import TechnicalAgent
from app.agents.news_agent import NewsAgent
from app.agents.ai_reasoning import ai_reasoning_service

logger = logging.getLogger(__name__)


class MarketAgent:
    """Orchestrates all agents and combines results"""
    
    def __init__(self):
        self.fundamental_agent = FundamentalAgent()
        self.technical_agent = TechnicalAgent()
        self.news_agent = NewsAgent()
    
    def fetch_market_context(self) -> Dict[str, Any]:
        """Fetch market-wide context"""
        try:
            # TODO: Fetch market indices, VIX, sector performance
            return {
                'market_mood': 0.5,
                'sector_performance': {},
                'created_at': None
            }
        except Exception as e:
            logger.error(f"Error fetching market context: {e}")
            return {'market_mood': 0.5}
    
    def analyze_symbol(self, symbol: str) -> FinalRecommendation:
        """Analyze a single symbol using all agents"""
        try:
            agent_results = self._run_agents_parallel(symbol)
            market_context = self.fetch_market_context()
            
            recommendation = self._calculate_consensus(
                agent_results, 
                market_context
            )
            
            return recommendation
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            raise
    
    def analyze_batch(self, symbols: List[str]) -> Dict[str, FinalRecommendation]:
        """Analyze multiple symbols in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.analyze_symbol, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
        
        return results
    
    def _run_agents_parallel(self, symbol: str) -> Dict[str, AgentResult]:
        """Run all agents in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.fundamental_agent.analyze, symbol): 'fundamental',
                executor.submit(self.technical_agent.analyze, symbol): 'technical',
                executor.submit(self.news_agent.analyze, symbol): 'news'
            }
            
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    results[agent_name] = future.result()
                except Exception as e:
                    logger.error(f"{agent_name} agent failed: {e}")
        
        return results
    
    def _calculate_consensus(
        self, 
        agent_results: Dict[str, AgentResult],
        market_context: Dict[str, Any]
    ) -> FinalRecommendation:
        """Calculate final recommendation using weighted voting"""
        weighted_votes = self._calculate_weighted_votes(agent_results)
        recommendation = max(weighted_votes, key=weighted_votes.get)
        avg_confidence = self._calculate_avg_confidence(agent_results)
        
        recommendation = self._apply_market_adjustment(
            recommendation, avg_confidence, market_context
        )
        
        return self._build_final_recommendation(
            agent_results, recommendation, avg_confidence,
            weighted_votes, market_context
        )
    
    def _build_final_recommendation(
        self, agent_results: Dict, recommendation: str,
        avg_confidence: float, weighted_votes: Dict,
        market_context: Dict
    ) -> FinalRecommendation:
        """Build final recommendation object"""
        symbol = self._extract_symbol(agent_results)
        reasoning = self._generate_reasoning(
            symbol, agent_results, recommendation,
            weighted_votes, market_context
        )
        
        final_rec = self._create_recommendation(
            symbol, recommendation, avg_confidence,
            reasoning, agent_results, weighted_votes
        )
        
        # Calculate trade details
        trade_details = self._calculate_trade_details(final_rec)
        final_rec.trade_details = trade_details
        
        return final_rec
    
    def _calculate_trade_details(
        self, recommendation: FinalRecommendation
    ) -> Dict[str, Any]:
        """Calculate trade details for recommendation"""
        try:
            from app.agents.trade_details_calculator import agent_trade_details_calculator
            return agent_trade_details_calculator.calculate_trade_details(
                recommendation
            )
        except Exception as e:
            logger.error(f"Error calculating trade details: {e}")
            return {}
    
    def _extract_symbol(self, agent_results: Dict) -> str:
        """Extract symbol from agent results"""
        return (
            list(agent_results.values())[0].symbol 
            if agent_results else ""
        )
    
    def _create_recommendation(
        self, symbol: str, recommendation: str,
        avg_confidence: float, reasoning: str,
        agent_results: Dict, weighted_votes: Dict
    ) -> FinalRecommendation:
        """Create final recommendation object"""
        return FinalRecommendation(
            symbol=symbol,
            recommendation=Verdict[recommendation],
            confidence=min(avg_confidence, 1.0),
            reasoning=reasoning,
            agent_breakdown=agent_results,
            weighted_votes=weighted_votes
        )
    
    def _calculate_weighted_votes(
        self, agent_results: Dict[str, AgentResult]
    ) -> Dict[str, float]:
        """Calculate weighted votes from agent results"""
        weighted_votes = {
            'BUY': 0.0, 'SELL': 0.0,
            'WATCH': 0.0, 'NEUTRAL': 0.0
        }
        
        for result in agent_results.values():
            weight = result.confidence * result.score
            weighted_votes[result.verdict.value] += weight
        
        return weighted_votes
    
    def _calculate_avg_confidence(
        self, agent_results: Dict[str, AgentResult]
    ) -> float:
        """Calculate average confidence from agent results"""
        if not agent_results:
            return 0.5
        
        total = sum(
            r.confidence * r.score for r in agent_results.values()
        )
        return total / len(agent_results)
    
    def _apply_market_adjustment(
        self,
        recommendation: str,
        confidence: float,
        market_context: Dict[str, Any]
    ) -> str:
        """Apply market context adjustments"""
        market_mood = market_context.get('market_mood', 0.5)
        
        if market_mood < 0.3 and recommendation == 'BUY':
            if confidence < 0.8:
                return 'WATCH'
        
        if market_mood > 0.7 and recommendation == 'WATCH':
            if confidence > 0.6:
                return 'BUY'
        
        return recommendation
    
    def _generate_reasoning(
        self,
        symbol: str,
        agent_results: Dict[str, AgentResult],
        recommendation: str,
        weighted_votes: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate AI-powered reasoning for recommendation"""
        try:
            return ai_reasoning_service.generate_reasoning(
                symbol=symbol,
                agent_results=agent_results,
                recommendation=Verdict[recommendation],
                weighted_votes=weighted_votes,
                market_context=market_context
            )
        except Exception as e:
            logger.error(f"AI reasoning failed: {e}")
            return self._generate_fallback_reasoning(
                agent_results, recommendation, weighted_votes
            )
    
    def _generate_fallback_reasoning(
        self,
        agent_results: Dict[str, AgentResult],
        recommendation: str,
        weighted_votes: Dict[str, float]
    ) -> str:
        """Generate fallback reasoning without AI"""
        parts = []
        
        for agent_name, result in agent_results.items():
            if result.verdict.value == recommendation:
                parts.append(
                    f"{agent_name.capitalize()} agent supports "
                    f"{recommendation} (score: {result.score:.2f})"
                )
        
        if not parts:
            parts.append(f"Weighted voting favors {recommendation}")
        
        return ". ".join(parts)

