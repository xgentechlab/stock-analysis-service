"""
Position Service
Converts recommendations to positions and creates position snapshots
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import re

from app.db.firestore_client import firestore_client
from app.agents.models import FinalRecommendation, AgentResult, Verdict
from app.models.schemas import (
    SetupType, Conviction, MonitoringFrequency, PositionState
)
from app.services.stocks import stocks_service

logger = logging.getLogger(__name__)


class PositionService:
    """Service for creating positions from recommendations"""
    
    def convert_recommendation_to_position(
        self, 
        recommendation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Convert a recommendation to a position with snapshot
        Returns position_id and snapshot_id
        """
        try:
            # Get recommendation
            recommendation = firestore_client.get_recommendation(recommendation_id)
            if not recommendation:
                raise ValueError(f"Recommendation {recommendation_id} not found")
            
            symbol = recommendation.get('symbol')
            
            # Get agent results from stock_analyses
            agent_results = self._get_agent_results(symbol)
            
            # Extract position data
            position_data = self._extract_position_data(
                recommendation, agent_results, user_id
            )
            
            # Create position snapshot
            snapshot_data = self._create_snapshot_data(
                recommendation, agent_results, position_data['id']
            )
            snapshot_id = firestore_client.create_position_snapshot(snapshot_data)
            
            # Link snapshot to position
            position_data['snapshotId'] = snapshot_id
            
            # Create position
            position_id = firestore_client.create_position(position_data)
            
            logger.info(
                f"Created position {position_id} with snapshot {snapshot_id} "
                f"for {symbol}"
            )
            
            return {
                'position_id': position_id,
                'snapshot_id': snapshot_id,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"Failed to convert recommendation to position: {e}")
            raise
    
    def _get_agent_results(self, symbol: str) -> Dict[str, AgentResult]:
        """Get agent results from stock_analyses collection"""
        try:
            from app.agents.storage import agent_storage
            from datetime import datetime
            
            # Get latest agent results (returns Dict)
            technical_dict = agent_storage.get_agent_result(symbol, 'technical')
            fundamental_dict = agent_storage.get_agent_result(symbol, 'fundamental')
            news_dict = agent_storage.get_agent_result(symbol, 'news')
            
            results = {}
            
            # Convert Dict to AgentResult objects
            if technical_dict:
                try:
                    technical_result = AgentResult(
                        agent_name=technical_dict.get('agent_name', 'technical'),
                        symbol=technical_dict.get('symbol', symbol),
                        score=technical_dict.get('score', 0.5),
                        verdict=Verdict(technical_dict.get('verdict', 'NEUTRAL')),
                        confidence=technical_dict.get('confidence', 0.5),
                        key_factors=technical_dict.get('key_factors', []),
                        metrics=technical_dict.get('metrics', {}),
                        analysis=technical_dict.get('analysis', {}),
                        ai_explanation=technical_dict.get('ai_explanation'),
                        created_at=datetime.fromisoformat(technical_dict.get('created_at', datetime.now(timezone.utc).isoformat())),
                        expires_at=datetime.fromisoformat(technical_dict['expires_at']) if technical_dict.get('expires_at') else None
                    )
                    results['technical'] = technical_result
                except Exception as e:
                    logger.warning(f"Could not convert technical result: {e}")
            
            if fundamental_dict:
                try:
                    fundamental_result = AgentResult(
                        agent_name=fundamental_dict.get('agent_name', 'fundamental'),
                        symbol=fundamental_dict.get('symbol', symbol),
                        score=fundamental_dict.get('score', 0.5),
                        verdict=Verdict(fundamental_dict.get('verdict', 'NEUTRAL')),
                        confidence=fundamental_dict.get('confidence', 0.5),
                        key_factors=fundamental_dict.get('key_factors', []),
                        metrics=fundamental_dict.get('metrics', {}),
                        analysis=fundamental_dict.get('analysis', {}),
                        ai_explanation=fundamental_dict.get('ai_explanation'),
                        created_at=datetime.fromisoformat(fundamental_dict.get('created_at', datetime.now(timezone.utc).isoformat())),
                        expires_at=datetime.fromisoformat(fundamental_dict['expires_at']) if fundamental_dict.get('expires_at') else None
                    )
                    results['fundamental'] = fundamental_result
                except Exception as e:
                    logger.warning(f"Could not convert fundamental result: {e}")
            
            if news_dict:
                try:
                    news_result = AgentResult(
                        agent_name=news_dict.get('agent_name', 'news'),
                        symbol=news_dict.get('symbol', symbol),
                        score=news_dict.get('score', 0.5),
                        verdict=Verdict(news_dict.get('verdict', 'NEUTRAL')),
                        confidence=news_dict.get('confidence', 0.5),
                        key_factors=news_dict.get('key_factors', []),
                        metrics=news_dict.get('metrics', {}),
                        analysis=news_dict.get('analysis', {}),
                        ai_explanation=news_dict.get('ai_explanation'),
                        created_at=datetime.fromisoformat(news_dict.get('created_at', datetime.now(timezone.utc).isoformat())),
                        expires_at=datetime.fromisoformat(news_dict['expires_at']) if news_dict.get('expires_at') else None
                    )
                    results['news'] = news_result
                except Exception as e:
                    logger.warning(f"Could not convert news result: {e}")
            
            return results
        except Exception as e:
            logger.warning(f"Could not fetch all agent results: {e}")
            return {}
    
    def _extract_position_data(
        self,
        recommendation: Dict[str, Any],
        agent_results: Dict[str, AgentResult],
        user_id: str
    ) -> Dict[str, Any]:
        """Extract all position data from recommendation and agent results"""
        symbol = recommendation.get('symbol')
        trade_details = recommendation.get('trade_details', {})
        entry_details = trade_details.get('entry', {})
        exit_details = trade_details.get('exit', {})
        position_sizing = trade_details.get('position_sizing', {})
        risk_metrics = trade_details.get('risk_metrics', {})
        
        # Basic data
        entry_price = entry_details.get('target_entry', 0.0)
        quantity = position_sizing.get('recommended_shares', 0)
        target_price = exit_details.get('target_exit', entry_price * 1.1)
        stop_loss = exit_details.get('stop_loss', entry_price * 0.95)
        
        # Extract computed fields
        setup_type = self._extract_setup_type(agent_results, recommendation)
        key_level, key_level_type = self._extract_key_level(
            agent_results, recommendation.get('action', 'buy')
        )
        catalyst = self._extract_catalyst(recommendation, agent_results)
        conviction = self._calculate_conviction(recommendation.get('confidence', 0.5))
        expected_timeline = self._calculate_expected_timeline(
            setup_type, risk_metrics.get('risk_reward_ratio', 1.0)
        )
        monitoring_frequency = self._calculate_monitoring_frequency(
            conviction, setup_type
        )
        
        now = datetime.now(timezone.utc).isoformat()
        
        return {
            'id': '',  # Will be set by Firestore
            'userId': user_id,
            'createdAt': now,
            'updatedAt': now,
            'symbol': symbol,
            'entryPrice': entry_price,
            'quantity': quantity,
            'entryDate': now,
            'targetPrice': target_price,
            'stopLoss': stop_loss,
            'expectedTimeline': expected_timeline,
            'setupType': setup_type.value,
            'keyLevel': key_level,
            'keyLevelType': key_level_type,
            'catalyst': catalyst,
            'conviction': conviction.value,
            'status': 'ACTIVE',
            'state': PositionState.WATCHING.value,
            'currentPrice': entry_price,
            'priceUpdatedAt': now,
            'unrealizedPnL': 0.0,
            'unrealizedPnLPercent': 0.0,
            'daysHeld': 0,
            'exitPrice': None,
            'exitDate': None,
            'exitReason': None,
            'realizedPnL': None,
            'monitoringFrequency': monitoring_frequency.value,
            'lastChecked': now,
            'alertsCount': 0,
            'snapshotId': ''  # Will be set after snapshot creation
        }
    
    def _extract_setup_type(
        self,
        agent_results: Dict[str, AgentResult],
        recommendation: Dict[str, Any]
    ) -> SetupType:
        """Classify setup type from agent results"""
        technical = agent_results.get('technical')
        news = agent_results.get('news')
        fundamental = agent_results.get('fundamental')
        
        action = recommendation.get('action', 'buy').lower()
        
        # Only classify for BUY actions
        if action != 'buy':
            return SetupType.MOMENTUM
        
        # 1. Check for BREAKOUT
        if technical:
            verdict = technical.verdict
            score = technical.score
            metrics = technical.metrics
            
            if verdict == Verdict.BUY and score > 0.7:
                current_price = metrics.get('close', 0)
                resistance_level = metrics.get('resistance_level', 0)
                
                # Check if price broke resistance
                if current_price > 0 and resistance_level > 0:
                    if current_price >= resistance_level * 0.98:  # Within 2% of resistance
                        return SetupType.BREAKOUT
        
        # 2. Check for NEWS_DRIVEN
        if news:
            if news.verdict == Verdict.BUY and news.score > 0.75:
                return SetupType.NEWS_DRIVEN
        
        # 3. Check for VALUE
        if fundamental:
            if fundamental.verdict == Verdict.BUY and fundamental.score > 0.7:
                return SetupType.VALUE
        
        # 4. Check for MOMENTUM
        if technical:
            metrics = technical.metrics
            macd = metrics.get('macd', 0)
            macd_signal = metrics.get('macd_signal', 0)
            rsi = metrics.get('rsi_14', 50)
            
            if macd > macd_signal and rsi > 50:
                return SetupType.MOMENTUM
        
        # Default
        return SetupType.MOMENTUM
    
    def _extract_key_level(
        self,
        agent_results: Dict[str, AgentResult],
        action: str
    ) -> tuple[float, str]:
        """Extract key level from technical metrics"""
        technical = agent_results.get('technical')
        
        if not technical:
            # Fallback values
            return (0.0, 'RESISTANCE' if action == 'buy' else 'SUPPORT')
        
        metrics = technical.metrics
        current_price = metrics.get('close', 0)
        
        if action.lower() == 'buy':
            # For BUY: use resistance level
            resistance_level = metrics.get('resistance_level', 0)
            recent_high = metrics.get('recent_high', 0)
            
            if resistance_level > 0:
                return (resistance_level, 'RESISTANCE')
            elif recent_high > 0:
                return (recent_high, 'RESISTANCE')
            else:
                return (current_price * 1.05, 'RESISTANCE')
        else:
            # For SELL: use support level
            support_level = metrics.get('support_level', 0)
            recent_low = metrics.get('recent_low', 0)
            
            if support_level > 0:
                return (support_level, 'SUPPORT')
            elif recent_low > 0:
                return (recent_low, 'SUPPORT')
            else:
                return (current_price * 0.95, 'SUPPORT')
    
    def _extract_catalyst(
        self,
        recommendation: Dict[str, Any],
        agent_results: Dict[str, AgentResult]
    ) -> str:
        """Extract catalyst from reasoning or key factors"""
        # 1. Try news agent first
        news = agent_results.get('news')
        if news and news.key_factors:
            # Get first news headline or key factor
            news_items = news.metrics.get('news_items', [])
            if news_items and len(news_items) > 0:
                headline = news_items[0].get('title', '')
                if headline:
                    return f"Recent news: {headline[:100]}"
            # Fallback to first key factor
            if news.key_factors:
                return f"News catalyst: {news.key_factors[0]}"
        
        # 2. Extract from AI reasoning
        reasoning = recommendation.get('reason', '')
        if reasoning:
            # Look for key phrases
            patterns = [
                r'earnings.*beat',
                r'Q\d.*earnings',
                r'analyst.*upgrade',
                r'announcement',
                r'partnership',
                r'expansion'
            ]
            for pattern in patterns:
                match = re.search(pattern, reasoning, re.IGNORECASE)
                if match:
                    # Extract sentence containing the match
                    sentences = reasoning.split('.')
                    for sentence in sentences:
                        if pattern.replace('.*', '').lower() in sentence.lower():
                            return sentence.strip()[:150]
        
        # 3. Combine key factors from agents
        key_factors = []
        for agent_name, result in agent_results.items():
            if result and result.key_factors:
                key_factors.extend(result.key_factors[:2])  # Top 2 from each
        
        if key_factors:
            return ' + '.join(key_factors[:3])  # Top 3 combined
        
        # 4. Fallback to first sentence of reasoning
        if reasoning:
            first_sentence = reasoning.split('.')[0]
            return first_sentence.strip()[:150]
        
        return "Position entry based on agent analysis"
    
    def _calculate_conviction(self, confidence: float) -> Conviction:
        """Map confidence to conviction level"""
        if confidence >= 0.8:
            return Conviction.HIGH
        elif confidence >= 0.6:
            return Conviction.MEDIUM
        else:
            return Conviction.LOW
    
    def _calculate_expected_timeline(
        self,
        setup_type: SetupType,
        risk_reward_ratio: float
    ) -> str:
        """Calculate expected timeline based on setup type and risk/reward"""
        if setup_type == SetupType.BREAKOUT:
            if risk_reward_ratio >= 2.0:
                return '3-5 days'
            elif risk_reward_ratio >= 1.5:
                return '5-7 days'
            else:
                return '7-10 days'
        
        elif setup_type == SetupType.MOMENTUM:
            if risk_reward_ratio >= 2.0:
                return '7-10 days'
            elif risk_reward_ratio >= 1.5:
                return '10-14 days'
            else:
                return '14-21 days'
        
        elif setup_type == SetupType.VALUE:
            if risk_reward_ratio >= 2.0:
                return '14-21 days'
            elif risk_reward_ratio >= 1.5:
                return '21-30 days'
            else:
                return '30-45 days'
        
        elif setup_type == SetupType.NEWS_DRIVEN:
            if risk_reward_ratio >= 2.0:
                return '1-3 days'
            elif risk_reward_ratio >= 1.5:
                return '3-5 days'
            else:
                return '5-7 days'
        
        return '7-14 days'  # Default
    
    def _calculate_monitoring_frequency(
        self,
        conviction: Conviction,
        setup_type: SetupType
    ) -> MonitoringFrequency:
        """Calculate monitoring frequency based on conviction and setup type"""
        if conviction == Conviction.HIGH:
            if setup_type == SetupType.BREAKOUT:
                return MonitoringFrequency.FIVEMIN
            else:
                return MonitoringFrequency.FIFTEENMIN
        elif conviction == Conviction.MEDIUM:
            return MonitoringFrequency.ONEHOUR
        else:
            return MonitoringFrequency.FOURHOUR
    
    def _create_snapshot_data(
        self,
        recommendation: Dict[str, Any],
        agent_results: Dict[str, AgentResult],
        position_id: str
    ) -> Dict[str, Any]:
        """Create position snapshot data"""
        technical = agent_results.get('technical')
        fundamental = agent_results.get('fundamental')
        
        # Technical data
        technical_data = self._extract_technical_snapshot(technical)
        
        # Fundamental data
        fundamental_data = self._extract_fundamental_snapshot(fundamental)
        
        # Market context
        market_data = self._fetch_market_context()
        
        # AI validation
        ai_data = self._extract_ai_validation(
            recommendation, agent_results, technical_data.get('trend', 'NEUTRAL')
        )
        
        return {
            'id': '',  # Will be set by Firestore
            'positionId': position_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'technical': technical_data,
            'fundamental': fundamental_data,
            'market': market_data,
            'ai': ai_data
        }
    
    def _extract_technical_snapshot(
        self,
        technical: Optional[AgentResult]
    ) -> Dict[str, Any]:
        """Extract technical snapshot data"""
        if not technical:
            return {
                'score': 0.5,
                'rsi': None,
                'volumeRatio': None,
                'sma20': None,
                'sma50': None,
                'trend': None,
                'support': [],
                'resistance': []
            }
        
        metrics = technical.metrics
        
        # Calculate volume ratio
        volume_ratio = None
        vol_today = metrics.get('vol_today', 0)
        vol_20 = metrics.get('vol20', 0)
        if vol_20 and vol_20 > 0:
            volume_ratio = vol_today / vol_20
        
        # Extract trend
        trend = 'NEUTRAL'
        sma20 = metrics.get('sma_20', 0)
        sma50 = metrics.get('sma_50', 0)
        close = metrics.get('close', 0)
        
        if close > 0 and sma20 > 0 and sma50 > 0:
            if close > sma20 > sma50:
                trend = 'BULLISH'
            elif close < sma20 < sma50:
                trend = 'BEARISH'
        
        # Extract support/resistance arrays
        support_levels = metrics.get('support_levels', {})
        resistance_levels = metrics.get('resistance_levels', {})
        
        support = [float(v) for v in support_levels.values()] if support_levels else []
        resistance = [float(v) for v in resistance_levels.values()] if resistance_levels else []
        
        return {
            'score': technical.score,
            'rsi': metrics.get('rsi_14') or metrics.get('rsi14'),
            'volumeRatio': volume_ratio,
            'sma20': sma20,
            'sma50': sma50,
            'trend': trend,
            'support': support,
            'resistance': resistance
        }
    
    def _extract_fundamental_snapshot(
        self,
        fundamental: Optional[AgentResult]
    ) -> Dict[str, Any]:
        """Extract fundamental snapshot data"""
        if not fundamental:
            return {
                'score': 0.5,
                'pe': None,
                'marketCap': None,
                'profitGrowth': None
            }
        
        metrics = fundamental.metrics
        
        # Extract from nested structure
        value_metrics = metrics.get('value_metrics', {})
        growth_metrics = metrics.get('growth_metrics', {})
        basic_fundamentals = metrics.get('basic_fundamentals', {})
        
        pe = value_metrics.get('pe_ratio') or basic_fundamentals.get('pe_ratio')
        market_cap = value_metrics.get('market_cap') or basic_fundamentals.get('market_cap_cr')
        profit_growth = growth_metrics.get('eps_growth_yoy') or growth_metrics.get('revenue_cagr')
        
        return {
            'score': fundamental.score,
            'pe': pe,
            'marketCap': market_cap,
            'profitGrowth': profit_growth
        }
    
    def _fetch_market_context(self) -> Dict[str, Any]:
        """Fetch market context (Nifty, VIX)"""
        try:
            import yfinance as yf
            
            # Fetch Nifty 50
            nifty_ticker = yf.Ticker("^NSEI")
            nifty_data = nifty_ticker.history(period="2d")
            
            nifty = None
            nifty_change = None
            
            if not nifty_data.empty:
                current = nifty_data['Close'].iloc[-1]
                previous = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else current
                nifty = float(current)
                nifty_change = ((current - previous) / previous * 100) if previous > 0 else 0.0
            
            # Try to fetch VIX (India VIX may not be available)
            vix = None
            try:
                vix_ticker = yf.Ticker("^INDIAVIX")
                vix_data = vix_ticker.history(period="1d")
                if not vix_data.empty:
                    vix = float(vix_data['Close'].iloc[-1])
            except:
                pass  # VIX not available
            
            return {
                'nifty': nifty,
                'niftyChange': nifty_change,
                'vix': vix
            }
        except Exception as e:
            logger.warning(f"Failed to fetch market context: {e}")
            return {
                'nifty': None,
                'niftyChange': None,
                'vix': None
            }
    
    def _extract_ai_validation(
        self,
        recommendation: Dict[str, Any],
        agent_results: Dict[str, AgentResult],
        trend: str
    ) -> Dict[str, Any]:
        """Extract AI validation data"""
        confidence = recommendation.get('confidence', 0.5)
        
        # Calculate overall score from weighted votes or average
        final_score = recommendation.get('final_score', 0.5)
        
        # Derive setup quality
        setup_quality = 'FAIR'
        if confidence >= 0.8:
            setup_quality = 'EXCELLENT'
        elif confidence >= 0.7:
            setup_quality = 'GOOD'
        
        # Extract concerns and strengths from AI explanations
        concerns = []
        strengths = []
        
        for agent_name, result in agent_results.items():
            if result and result.ai_explanation:
                explanation = result.ai_explanation.lower()
                
                # Look for negative keywords
                negative_keywords = ['risk', 'concern', 'caution', 'weak', 'decline', 'negative']
                if any(kw in explanation for kw in negative_keywords):
                    # Extract relevant sentence
                    sentences = result.ai_explanation.split('.')
                    for sentence in sentences:
                        if any(kw in sentence.lower() for kw in negative_keywords):
                            concerns.append(sentence.strip()[:100])
                
                # Look for positive keywords
                positive_keywords = ['strong', 'growth', 'positive', 'bullish', 'opportunity', 'momentum']
                if any(kw in explanation for kw in positive_keywords):
                    sentences = result.ai_explanation.split('.')
                    for sentence in sentences:
                        if any(kw in sentence.lower() for kw in positive_keywords):
                            strengths.append(sentence.strip()[:100])
        
        # Limit to top 3 each
        concerns = concerns[:3]
        strengths = strengths[:3]
        
        return {
            'overallScore': final_score,
            'confidence': confidence,
            'setupQuality': setup_quality,
            'concerns': concerns,
            'strengths': strengths,
            'winRate': 0.5  # Default, will be calculated from historical data
        }


# Singleton instance
position_service = PositionService()

