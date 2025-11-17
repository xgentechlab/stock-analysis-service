"""
Storage layer for agent results
"""
from typing import Dict, Optional, Any
from datetime import datetime
import logging

from app.agents.models import AgentResult, FinalRecommendation
from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)


class AgentStorage:
    """Handles storage and retrieval of agent results"""
    
    def store_agent_result(self, result: AgentResult) -> bool:
        """Store individual agent result"""
        try:
            doc_ref = firestore_client.db.collection(
                "stock_analyses"
            ).document(result.symbol)
            
            field_name = f"{result.agent_name}_analysis"
            data = result.to_dict()
            
            doc_ref.set({field_name: data}, merge=True)
            return True
        except Exception as e:
            logger.error(f"Error storing agent result: {e}")
            return False
    
    def store_final_recommendation(
        self, 
        recommendation: FinalRecommendation,
        route_to_recommendations: bool = True,
        user_id: str = "default_user"
    ) -> bool:
        """Store final recommendation and optionally route to recommendations/watchlists"""
        try:
            # Store in stock_analyses collection
            doc_ref = firestore_client.db.collection(
                "stock_analyses"
            ).document(recommendation.symbol)
            
            data = recommendation.to_dict()
            doc_ref.set({"final_recommendation": data}, merge=True)
            
            # Route to recommendations/watchlists if requested
            if route_to_recommendations:
                try:
                    from app.agents.recommendation_router import route_agent_recommendation
                    route_agent_recommendation(recommendation, user_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to route recommendation for {recommendation.symbol}: {e}"
                    )
            
            return True
        except Exception as e:
            logger.error(f"Error storing recommendation: {e}")
            return False
    
    def store_market_context(self, context: Dict[str, Any]) -> bool:
        """Store market context"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            doc_ref = firestore_client.db.collection(
                "market_context"
            ).document(today)
            
            context['created_at'] = datetime.utcnow().isoformat()
            doc_ref.set(context)
            return True
        except Exception as e:
            logger.error(f"Error storing market context: {e}")
            return False
    
    def get_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get full analysis for symbol"""
        try:
            doc_ref = firestore_client.db.collection(
                "stock_analyses"
            ).document(symbol)
            
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting analysis: {e}")
            return None
    
    def get_agent_result(
        self, 
        symbol: str, 
        agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific agent result"""
        try:
            analysis = self.get_analysis(symbol)
            if not analysis:
                return None
            
            field_name = f"{agent_name}_analysis"
            return analysis.get(field_name)
        except Exception as e:
            logger.error(f"Error getting agent result: {e}")
            return None
    
    def is_agent_result_expired(
        self, 
        symbol: str, 
        agent_name: str
    ) -> bool:
        """Check if agent result is expired"""
        try:
            result = self.get_agent_result(symbol, agent_name)
            if not result:
                return True
            
            expires_at_str = result.get('expires_at')
            if not expires_at_str:
                return True
            
            expires_at = datetime.fromisoformat(expires_at_str)
            return datetime.utcnow() > expires_at
        except Exception as e:
            logger.error(f"Error checking expiry: {e}")
            return True


agent_storage = AgentStorage()

