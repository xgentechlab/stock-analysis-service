"""
AI Reasoning Service - Synthesizes agent results into plain English explanations
Uses OpenAI or Claude to generate contextual reasoning from agent outputs
"""
from typing import Dict, Any, Optional, List
import logging
import json

from app.config import settings
from app.agents.models import AgentResult, Verdict
from app.services.openai_client import openai_client
from app.services.claude_client import claude_client

logger = logging.getLogger(__name__)


class AIReasoningService:
    """Generates AI-powered explanations from agent results"""
    
    def __init__(self):
        self.provider = settings.ai_provider.lower()
        self.use_openai = self.provider == "openai"
        self.use_claude = self.provider == "claude"
    
    def generate_reasoning(
        self,
        symbol: str,
        agent_results: Dict[str, AgentResult],
        recommendation: Verdict,
        weighted_votes: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate AI-powered reasoning from agent results"""
        try:
            if self.use_openai:
                return self._generate_with_openai(
                    symbol, agent_results, recommendation,
                    weighted_votes, market_context
                )
            elif self.use_claude:
                return self._generate_with_claude(
                    symbol, agent_results, recommendation,
                    weighted_votes, market_context
                )
        except Exception as e:
            logger.error(f"AI reasoning failed: {e}")
        
        return self._generate_fallback_reasoning(
            agent_results, recommendation, weighted_votes
        )
    
    def _generate_with_openai(
        self, symbol: str, agent_results: Dict[str, AgentResult],
        recommendation: Verdict, weighted_votes: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate reasoning using OpenAI"""
        prompt = self._build_reasoning_prompt(
            symbol, agent_results, recommendation,
            weighted_votes, market_context
        )
        
        response = self._call_openai_api(prompt)
        return self._extract_reasoning(response)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with prompt"""
        if not openai_client.client:
            raise ValueError("OpenAI client not initialized")
        
        response = openai_client.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    def _generate_with_claude(
        self, symbol: str, agent_results: Dict[str, AgentResult],
        recommendation: Verdict, weighted_votes: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate reasoning using Claude"""
        prompt = self._build_reasoning_prompt(
            symbol, agent_results, recommendation,
            weighted_votes, market_context
        )
        
        response = self._call_claude_api(prompt)
        return self._extract_reasoning(response)
    
    def _call_claude_api(self, prompt: str) -> str:
        """Call Claude API with prompt"""
        if not claude_client.client:
            raise ValueError("Claude client not initialized")
        
        response = claude_client.client.messages.create(
            model=settings.claude_model,
            max_tokens=500,
            temperature=0.3,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    
    def _build_reasoning_prompt(
        self, symbol: str, agent_results: Dict[str, AgentResult],
        recommendation: Verdict, weighted_votes: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Build prompt for AI reasoning"""
        agent_summaries = self._format_agent_results(agent_results)
        vote_summary = self._format_weighted_votes(weighted_votes)
        header = self._build_prompt_header(symbol, recommendation)
        instructions = self._build_prompt_instructions(recommendation)
        
        return f"""{header}

AGENT ANALYSIS RESULTS:
{agent_summaries}

CONSENSUS VOTING:
{vote_summary}

MARKET CONTEXT:
Market Mood: {market_context.get('market_mood', 0.5):.2f}

{instructions}"""
    
    def _build_prompt_header(
        self, symbol: str, recommendation: Verdict
    ) -> str:
        """Build prompt header section"""
        return f"""Stock: {symbol}
Final Recommendation: {recommendation.value}"""
    
    def _build_prompt_instructions(self, recommendation: Verdict) -> str:
        """Build prompt instructions section"""
        return f"""Generate a clear, concise explanation (2-3 sentences) that:
1. Summarizes why the agents reached this {recommendation.value} recommendation
2. Highlights the most important factors supporting this decision
3. Mentions any conflicting signals if applicable
4. Uses plain English, avoiding technical jargon

Return only the explanation text, no JSON or formatting."""
    
    def _format_agent_results(
        self, agent_results: Dict[str, AgentResult]
    ) -> str:
        """Format agent results for prompt"""
        summaries = []
        
        for name, result in agent_results.items():
            factors = ", ".join(result.key_factors[:3])
            summaries.append(
                f"- {name.capitalize()}: {result.verdict.value} "
                f"(score: {result.score:.2f}, confidence: {result.confidence:.2f})"
                f"\n  Key factors: {factors}"
            )
        
        return "\n".join(summaries)
    
    def _format_weighted_votes(
        self, weighted_votes: Dict[str, float]
    ) -> str:
        """Format weighted votes for prompt"""
        votes = []
        for verdict, weight in sorted(
            weighted_votes.items(), key=lambda x: x[1], reverse=True
        ):
            if weight > 0:
                votes.append(f"- {verdict}: {weight:.2f}")
        
        return "\n".join(votes) if votes else "No votes"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for AI reasoning"""
        return """You are a financial analyst explaining investment decisions 
in simple, clear language. Translate technical analysis into plain English 
that any investor can understand. Focus on what matters for making money, 
not technical complexity."""
    
    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from AI response"""
        content = content.strip()
        
        if content.startswith('{'):
            try:
                parsed = json.loads(content)
                if 'reasoning' in parsed:
                    return parsed['reasoning']
                if 'explain' in parsed:
                    return parsed['explain']
            except json.JSONDecodeError:
                pass
        
        return content[:500]
    
    def _generate_fallback_reasoning(
        self, agent_results: Dict[str, AgentResult],
        recommendation: Verdict, weighted_votes: Dict[str, float]
    ) -> str:
        """Generate fallback reasoning without AI"""
        parts = []
        
        for name, result in agent_results.items():
            if result.verdict == recommendation:
                parts.append(
                    f"{name.capitalize()} agent supports {recommendation.value} "
                    f"(score: {result.score:.2f})"
                )
        
        if not parts:
            parts.append(f"Weighted voting favors {recommendation.value}")
        
        return ". ".join(parts)


# Singleton instance
ai_reasoning_service = AIReasoningService()

