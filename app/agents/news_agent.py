"""
News Agent - Analyzes news and sentiment
"""
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import yfinance as yf

from app.agents.base_agent import BaseAgent
from app.agents.models import Verdict

logger = logging.getLogger(__name__)


class NewsAgent(BaseAgent):
    """Agent for news and sentiment analysis"""
    
    def __init__(self):
        super().__init__("news", ttl_hours=1)  # 1 hour
        self.nse_suffix = ".NS"
    
    def fetch_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch news data from yfinance"""
        try:
            # Format symbol for yfinance
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
            
            ticker = yf.Ticker(ticker_symbol)
            news_items = ticker.news or []
            
            # Process news items
            processed_news = self._process_news_items(news_items)
            recent_count = len(processed_news)
            
            # Calculate sentiment
            sentiment_score = self._calculate_sentiment(processed_news)
            
            # Calculate final score and confidence
            final_score = self._calculate_news_score(sentiment_score, recent_count)
            confidence = self._calculate_base_confidence(recent_count, sentiment_score)
            
            return {
                'metrics': {
                    'recent_news_count': recent_count,
                    'sentiment_score': sentiment_score,
                    'news_items': processed_news
                },
                'analysis': {
                    'final_score': final_score,
                    'confidence': confidence
                },
                'news_items': processed_news
            }
        except Exception as e:
            logger.error(f"Error fetching news data for {symbol}: {e}")
            return {
                'metrics': {
                    'recent_news_count': 0,
                    'sentiment_score': 0.5
                },
                'analysis': {
                    'final_score': 0.5,
                    'confidence': 0.3
                },
                'news_items': []
            }
    
    def calculate_score(self, data: Dict[str, Any]) -> float:
        """Calculate news score"""
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
        """Calculate confidence"""
        analysis = data.get('analysis', {})
        base_confidence = float(analysis.get('confidence', 0.5))
        
        metrics = data.get('metrics', {})
        news_count = metrics.get('recent_news_count', 0)
        
        if news_count == 0:
            return 0.3  # Low confidence if no news
        
        return min(1.0, base_confidence * (1 + news_count / 10))
    
    def extract_key_factors(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> list:
        """Extract key factors"""
        factors = []
        metrics = data.get('metrics', {})
        
        sentiment = metrics.get('sentiment_score', 0.5)
        if sentiment > 0.7:
            factors.append("Positive news sentiment")
        elif sentiment < 0.3:
            factors.append("Negative news sentiment")
        
        news_count = metrics.get('recent_news_count', 0)
        if news_count > 5:
            factors.append(f"High news activity ({news_count} items)")
        
        return factors[:5]
    
    def _process_news_items(self, news_items: List[Dict]) -> List[Dict[str, Any]]:
        """Process raw news items from yfinance"""
        processed = []
        
        for item in news_items[:20]:  # Limit to 20 most recent
            try:
                content = item.get('content', {})
                if isinstance(content, dict):
                    processed.append({
                        'title': content.get('title', ''),
                        'summary': content.get('summary', ''),
                        'pub_date': content.get('pubDate', ''),
                        'provider': content.get('provider', {}).get('displayName', 'Unknown'),
                        'url': content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                    })
            except Exception as e:
                logger.warning(f"Error processing news item: {e}")
                continue
        
        return processed
    
    def _calculate_sentiment(self, news_items: List[Dict[str, Any]]) -> float:
        """Calculate sentiment score from news items"""
        if not news_items:
            return 0.5  # Neutral
        
        # Simple keyword-based sentiment analysis
        positive_keywords = [
            'profit', 'growth', 'gain', 'rise', 'up', 'strong', 'beat',
            'positive', 'bullish', 'upgrade', 'buy', 'outperform', 'success',
            'expansion', 'acquisition', 'partnership', 'launch', 'record'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'fall', 'down', 'weak', 'miss', 'negative',
            'bearish', 'downgrade', 'sell', 'underperform', 'failure',
            'contraction', 'layoff', 'bankruptcy', 'warning', 'concern'
        ]
        
        total_score = 0.0
        valid_items = 0
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            
            positive_count = sum(1 for kw in positive_keywords if kw in text)
            negative_count = sum(1 for kw in negative_keywords if kw in text)
            
            if positive_count + negative_count > 0:
                item_sentiment = (positive_count - negative_count) / (positive_count + negative_count + 1)
                # Normalize to 0-1 range
                item_sentiment = (item_sentiment + 1) / 2
                total_score += item_sentiment
                valid_items += 1
        
        if valid_items == 0:
            return 0.5
        
        return total_score / valid_items
    
    def _calculate_news_score(
        self, sentiment_score: float, news_count: int
    ) -> float:
        """Calculate final news score"""
        # Base score from sentiment
        base_score = sentiment_score
        
        # Adjust based on news activity
        if news_count == 0:
            return 0.5  # Neutral if no news
        elif news_count < 3:
            # Low activity - reduce confidence in sentiment
            return 0.4 + (base_score - 0.5) * 0.5
        elif news_count > 10:
            # High activity - amplify sentiment
            return 0.3 + (base_score - 0.5) * 1.4
        
        return base_score
    
    def _calculate_base_confidence(
        self, news_count: int, sentiment_score: float
    ) -> float:
        """Calculate base confidence from news data"""
        if news_count == 0:
            return 0.3
        
        # Higher confidence with more news and clearer sentiment
        activity_factor = min(1.0, news_count / 10)
        sentiment_clarity = abs(sentiment_score - 0.5) * 2  # 0-1, higher = clearer
        
        return 0.5 + (activity_factor * sentiment_clarity * 0.3)

