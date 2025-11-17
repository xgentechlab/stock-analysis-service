"""
Technical Agent - Analyzes price action and technical indicators
"""
from typing import Dict, Any, Optional
import logging

from app.agents.base_agent import BaseAgent
from app.agents.models import Verdict, AgentResult
from app.services.stocks import stocks_service
from app.services.indicators import calculate_technical_snapshot
from app.services.enhanced_scoring import enhanced_scoring

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    """Agent for technical analysis"""
    
    def __init__(self):
        super().__init__("technical", ttl_hours=24)  # 1 day
    
    def fetch_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch technical data"""
        try:
            ticker_data = stocks_service._fetch_all_data_optimized(symbol)
            if not ticker_data:
                return {}
            
            ohlcv = ticker_data.get("ohlcv_60d")
            if ohlcv is None or ohlcv.empty:
                return {}
            
            return self._process_technical_data(ohlcv)
        except Exception as e:
            logger.error(f"Error fetching technical data: {e}")
            return {}
    
    def _process_technical_data(self, ohlcv) -> Dict[str, Any]:
        """Process OHLCV data into technical metrics"""
        technical_snapshot = calculate_technical_snapshot(ohlcv)
        score_data = enhanced_scoring.calculate_enhanced_score(
            technical_snapshot
        )
        
        # Calculate support and resistance levels
        support_resistance = self._calculate_support_resistance(
            ohlcv, technical_snapshot
        )
        
        # Add support/resistance to metrics
        technical_snapshot.update(support_resistance)
        
        return {
            'metrics': technical_snapshot,
            'analysis': score_data,
            'ohlcv': ohlcv
        }
    
    def _calculate_support_resistance(
        self, ohlcv, technical_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            import pandas as pd
            
            current_price = technical_snapshot.get('close', 0)
            if current_price == 0:
                return {}
            
            high = ohlcv['High']
            low = ohlcv['Low']
            close = ohlcv['Close']
            volume = ohlcv['Volume']
            
            # Calculate support levels (below current price)
            sma_20 = technical_snapshot.get('sma_20', 0)
            sma_50 = technical_snapshot.get('sma_50', 0)
            vwap = technical_snapshot.get('vwap', 0)
            recent_low = low.tail(20).min() if len(low) >= 20 else low.min()
            atr_14 = technical_snapshot.get('atr_14', 0)
            
            # Support: Use VWAP, SMA20, SMA50, or recent low (whichever is below price)
            support_levels = []
            if vwap and vwap < current_price:
                support_levels.append(('vwap', vwap))
            if sma_20 and sma_20 < current_price:
                support_levels.append(('sma_20', sma_20))
            if sma_50 and sma_50 < current_price:
                support_levels.append(('sma_50', sma_50))
            if recent_low and recent_low < current_price:
                # Validate: not more than 10% below
                if recent_low >= current_price * 0.9:
                    support_levels.append(('recent_low', recent_low))
            
            # Primary support: closest level below current price
            primary_support = max(
                [level for _, level in support_levels],
                default=current_price * 0.95
            )
            
            # ATR-based stop loss (2x ATR below current price)
            atr_stop_loss = current_price - (2 * atr_14) if atr_14 > 0 else None
            
            # Calculate resistance levels (above current price)
            recent_high = high.tail(20).max() if len(high) >= 20 else high.max()
            high_52w = high.tail(252).max() if len(high) >= 252 else high.max()
            bb_upper = technical_snapshot.get('bb_upper', None)
            
            resistance_levels = []
            if recent_high and recent_high > current_price:
                resistance_levels.append(('recent_high', recent_high))
            if high_52w and high_52w > current_price:
                resistance_levels.append(('high_52w', high_52w))
            if bb_upper and bb_upper > current_price:
                resistance_levels.append(('bb_upper', bb_upper))
            
            # Primary resistance: closest level above current price
            primary_resistance = min(
                [level for _, level in resistance_levels],
                default=current_price * 1.10
            )
            
            return {
                'support_level': primary_support,
                'resistance_level': primary_resistance,
                'atr_stop_loss': atr_stop_loss,
                'support_levels': {name: level for name, level in support_levels},
                'resistance_levels': {name: level for name, level in resistance_levels}
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def calculate_score(self, data: Dict[str, Any]) -> float:
        """Calculate technical score"""
        analysis = data.get('analysis', {})
        return float(analysis.get('final_score', 0.5))
    
    def generate_verdict(self, score: float) -> Verdict:
        """Generate verdict from score"""
        if score >= 0.65:
            return Verdict.BUY
        elif score <= 0.35:
            return Verdict.SELL
        else:
            return Verdict.WATCH
    
    def calculate_confidence(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> float:
        """Calculate confidence"""
        analysis = data.get('analysis', {})
        base_confidence = float(analysis.get('confidence', 0.5))
        
        metrics = data.get('metrics', {})
        indicator_count = sum(
            1 for k in ['rsi_14', 'macd', 'sma_20', 'volume_20d_avg']
            if metrics.get(k) is not None
        )
        data_quality = indicator_count / 4.0
        
        return min(1.0, base_confidence * data_quality)
    
    def extract_key_factors(
        self, 
        data: Dict[str, Any], 
        score: float
    ) -> list:
        """Extract key factors"""
        factors = []
        metrics = data.get('metrics', {})
        
        factors.extend(self._extract_rsi_factors(metrics))
        factors.extend(self._extract_macd_factors(metrics))
        factors.extend(self._extract_price_factors(metrics))
        
        return factors[:5]
    
    def _extract_rsi_factors(self, metrics: Dict) -> list:
        """Extract RSI-based factors"""
        factors = []
        rsi = metrics.get('rsi_14')
        if rsi:
            if rsi < 30:
                factors.append("Oversold condition (RSI < 30)")
            elif rsi > 70:
                factors.append("Overbought condition (RSI > 70)")
        return factors
    
    def _extract_macd_factors(self, metrics: Dict) -> list:
        """Extract MACD-based factors"""
        factors = []
        macd = metrics.get('macd')
        macd_signal = metrics.get('macd_signal')
        if macd and macd_signal and macd > macd_signal:
            factors.append("Bullish MACD crossover")
        return factors
    
    def _extract_price_factors(self, metrics: Dict) -> list:
        """Extract price-based factors"""
        factors = []
        price_change = metrics.get('price_change_5d_pct', 0)
        if abs(price_change) > 5:
            factors.append(
                f"Significant 5-day move ({price_change:.1f}%)"
            )
        return factors
    
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
            return agent_ai_reasoning.generate_technical_explanation(
                symbol, metrics, score, verdict.value
            )
        except Exception as e:
            logger.warning(f"AI explanation generation failed: {e}")
            return None

