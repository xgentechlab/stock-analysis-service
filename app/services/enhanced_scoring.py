"""
Enhanced Scoring System
Uses advanced technical indicators for improved signal generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedScoringEngine:
    """Enhanced scoring system using advanced technical analysis"""
    
    def __init__(self):
        # Scoring weights for different components
        self.weights = {
            "momentum": 0.30,      # Reduced from 0.45
            "volume": 0.25,        # Reduced from 0.35
            "breakout": 0.15,      # Reduced from 0.20
            "divergence": 0.15,    # New component
            "multi_timeframe": 0.15  # New component
        }
        
        # Thresholds for different signal strengths
        self.thresholds = {
            "strong_buy": 0.8,
            "buy": 0.6,
            "neutral": 0.4,
            "sell": 0.2,
            "strong_sell": 0.0
        }
    
    def calculate_enhanced_score(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate enhanced score using advanced technical indicators
        Returns comprehensive scoring breakdown
        """
        try:
            scores = {}
            
            # 1. Enhanced Momentum Score
            momentum_score = self._calculate_enhanced_momentum_score(technical_data)
            scores["momentum"] = momentum_score
            
            # 2. Enhanced Volume Score
            volume_score = self._calculate_enhanced_volume_score(technical_data)
            scores["volume"] = volume_score
            
            # 3. Enhanced Breakout Score
            breakout_score = self._calculate_enhanced_breakout_score(technical_data)
            scores["breakout"] = breakout_score
            
            # 4. Divergence Score (New)
            divergence_score = self._calculate_divergence_score(technical_data)
            scores["divergence"] = divergence_score
            
            # 5. Multi-timeframe Score (New)
            mtf_score = self._calculate_mtf_score(technical_data)
            scores["multi_timeframe"] = mtf_score
            
            # Calculate weighted final score
            final_score = sum(
                self.weights[component] * score 
                for component, score in scores.items()
            )
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(final_score)
            
            return {
                "final_score": final_score,
                "signal_strength": signal_strength,
                "component_scores": scores,
                "weights": self.weights,
                "confidence": self._calculate_confidence(scores, technical_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced score: {e}")
            return {"final_score": 0.0, "signal_strength": "neutral", "error": str(e)}
    
    def _calculate_enhanced_momentum_score(self, data: Dict[str, Any]) -> float:
        """Calculate enhanced momentum score using multiple indicators"""
        try:
            score_components = []
            
            # RSI momentum (0-1)
            rsi = data.get("rsi_14", 50)
            if rsi is not None:
                # RSI momentum: 0.5 at 50, 1.0 at 30 (oversold), 0.0 at 70 (overbought)
                rsi_momentum = max(0, min(1, (70 - rsi) / 40))
                score_components.append(rsi_momentum)
            
            # MACD momentum (0-1)
            macd = data.get("macd", 0)
            macd_signal = data.get("macd_signal", 0)
            macd_histogram = data.get("macd_histogram", 0)
            
            if macd is not None and macd_signal is not None:
                macd_momentum = 1.0 if macd > macd_signal else 0.0
                # Add histogram strength
                if macd_histogram is not None:
                    histogram_strength = min(1.0, abs(macd_histogram) * 1000)  # Scale histogram
                    macd_momentum = (macd_momentum + histogram_strength) / 2
                score_components.append(macd_momentum)
            
            # Stochastic RSI momentum (0-1)
            stoch_rsi_k = data.get("stoch_rsi_k", 50)
            if stoch_rsi_k is not None:
                # Similar to RSI: 0.5 at 50, 1.0 at 20 (oversold), 0.0 at 80 (overbought)
                stoch_momentum = max(0, min(1, (80 - stoch_rsi_k) / 60))
                score_components.append(stoch_momentum)
            
            # Williams %R momentum (0-1)
            williams_r = data.get("williams_r", -50)
            if williams_r is not None:
                # Williams %R: -20 to -80 range, convert to 0-1
                williams_momentum = max(0, min(1, (williams_r + 80) / 60))
                score_components.append(williams_momentum)
            
            # ROC momentum (0-1)
            roc_5 = data.get("roc_5", 0)
            roc_10 = data.get("roc_10", 0)
            roc_20 = data.get("roc_20", 0)
            
            roc_scores = []
            for roc in [roc_5, roc_10, roc_20]:
                if roc is not None:
                    # ROC: positive = bullish, negative = bearish
                    roc_momentum = max(0, min(1, (roc + 10) / 20))  # -10% to +10% range
                    roc_scores.append(roc_momentum)
            
            if roc_scores:
                score_components.append(float(np.mean(roc_scores)))
            
            # Price vs moving averages
            current_price = data.get("current_price", 0)
            sma_20 = data.get("sma_20")
            sma_50 = data.get("sma_50")
            
            if current_price and sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    score_components.append(1.0)  # Strong uptrend
                elif current_price > sma_20:
                    score_components.append(0.7)  # Moderate uptrend
                elif current_price > sma_50:
                    score_components.append(0.3)  # Weak uptrend
                else:
                    score_components.append(0.0)  # Downtrend
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating enhanced momentum score: {e}")
            return 0.5
    
    def _calculate_enhanced_volume_score(self, data: Dict[str, Any]) -> float:
        """Calculate enhanced volume score using volume profile analysis"""
        try:
            score_components = []
            
            # VWAP analysis
            current_price = data.get("current_price", 0)
            vwap = data.get("vwap", 0)
            vwap_upper = data.get("vwap_upper", 0)
            vwap_lower = data.get("vwap_lower", 0)
            
            if current_price and vwap:
                if current_price > vwap_upper:
                    score_components.append(1.0)  # Above upper VWAP band
                elif current_price > vwap:
                    score_components.append(0.7)  # Above VWAP
                elif current_price > vwap_lower:
                    score_components.append(0.3)  # Between VWAP bands
                else:
                    score_components.append(0.0)  # Below lower VWAP band
            
            # OBV trend analysis
            obv = data.get("obv", 0)
            if obv is not None:
                # This would need historical OBV data for trend analysis
                # For now, use current OBV value
                obv_score = min(1.0, max(0.0, obv / 1000000))  # Scale OBV
                score_components.append(obv_score)
            
            # A/D Line trend analysis
            ad_line = data.get("ad_line", 0)
            if ad_line is not None:
                # Similar to OBV, would need historical data for trend
                ad_score = min(1.0, max(0.0, ad_line / 1000000))  # Scale A/D Line
                score_components.append(ad_score)
            
            # Volume profile analysis
            volume_profile = data.get("volume_profile", {})
            high_volume_nodes = volume_profile.get("high_volume_nodes", [])
            
            if high_volume_nodes and current_price:
                # Check if current price is near high volume nodes (support/resistance)
                price_near_high_volume = False
                for node in high_volume_nodes:
                    node_price = node.get("price", 0)
                    if abs(current_price - node_price) / current_price < 0.02:  # Within 2%
                        price_near_high_volume = True
                        break
                
                if price_near_high_volume:
                    score_components.append(0.8)  # Near high volume node
                else:
                    score_components.append(0.4)  # Not near high volume node
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating enhanced volume score: {e}")
            return 0.5
    
    def _calculate_enhanced_breakout_score(self, data: Dict[str, Any]) -> float:
        """Calculate enhanced breakout score using multiple breakout indicators"""
        try:
            score_components = []
            
            # Traditional breakout (price > recent high)
            current_price = data.get("current_price", 0)
            # This would need historical high data - simplified for now
            breakout_signal = data.get("is_breakout", False)
            if breakout_signal:
                score_components.append(1.0)
            else:
                score_components.append(0.0)
            
            # ATR-based volatility
            atr_14 = data.get("atr_14", 0)
            if atr_14 and current_price:
                # Higher ATR relative to price indicates more volatility/breakout potential
                atr_ratio = atr_14 / current_price
                atr_score = min(1.0, atr_ratio * 20)  # Scale ATR ratio
                score_components.append(atr_score)
            
            # MACD histogram for momentum confirmation
            macd_histogram = data.get("macd_histogram", 0)
            if macd_histogram is not None:
                # Positive histogram indicates increasing momentum
                histogram_score = max(0, min(1, (macd_histogram + 0.01) * 50))
                score_components.append(histogram_score)
            
            # ROC for momentum confirmation
            roc_5 = data.get("roc_5", 0)
            if roc_5 is not None:
                # High positive ROC indicates strong momentum
                roc_score = max(0, min(1, (roc_5 + 5) / 10))  # -5% to +5% range
                score_components.append(roc_score)
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating enhanced breakout score: {e}")
            return 0.5
    
    def _calculate_divergence_score(self, data: Dict[str, Any]) -> float:
        """Calculate divergence score based on RSI and price divergences"""
        try:
            score = 0.5  # Neutral base score
            
            # RSI divergence analysis
            rsi_divergence = data.get("rsi_divergence", {})
            
            if rsi_divergence.get("bullish_divergence"):
                score = 0.9  # Strong bullish signal
            elif rsi_divergence.get("bearish_divergence"):
                score = 0.1  # Strong bearish signal
            
            # Additional divergence checks could be added here
            # (e.g., MACD divergence, volume divergence, etc.)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating divergence score: {e}")
            return 0.5
    
    def _calculate_mtf_score(self, data: Dict[str, Any]) -> float:
        """Calculate multi-timeframe score based on trend alignment"""
        try:
            score_components = []
            
            # Check trend alignment across timeframes
            timeframes = ["1m", "5m", "15m", "1d", "1wk"]
            bullish_count = 0
            total_timeframes = 0
            
            for tf in timeframes:
                trend = data.get(f"{tf}_trend")
                if trend:
                    total_timeframes += 1
                    if trend == "bullish":
                        bullish_count += 1
                    elif trend == "bearish":
                        bullish_count -= 1  # Count bearish as negative
            
            if total_timeframes > 0:
                # Score based on trend alignment
                alignment_ratio = (bullish_count + total_timeframes) / (2 * total_timeframes)
                score_components.append(alignment_ratio)
            
            # Check momentum alignment across timeframes
            momentum_scores = []
            for tf in timeframes:
                momentum = data.get(f"{tf}_momentum", 0)
                if momentum is not None:
                    # Convert momentum to 0-1 score
                    momentum_score = max(0, min(1, (momentum + 10) / 20))  # -10% to +10% range
                    momentum_scores.append(momentum_score)
            
            if momentum_scores:
                # Higher score if momentum is consistent across timeframes
                momentum_consistency = 1.0 - float(np.std(momentum_scores))
                score_components.append(momentum_consistency)
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe score: {e}")
            return 0.5
    
    def _determine_signal_strength(self, final_score: float) -> str:
        """Determine signal strength based on final score"""
        if final_score >= self.thresholds["strong_buy"]:
            return "strong_buy"
        elif final_score >= self.thresholds["buy"]:
            return "buy"
        elif final_score >= self.thresholds["neutral"]:
            return "neutral"
        elif final_score >= self.thresholds["sell"]:
            return "sell"
        else:
            return "strong_sell"
    
    def _calculate_confidence(self, scores: Dict[str, float], data: Dict[str, Any]) -> float:
        """Calculate confidence level based on score consistency and data quality"""
        try:
            # Score consistency (lower std = higher confidence)
            score_values = list(scores.values())
            consistency = 1.0 - float(np.std(score_values))
            
            # Data quality indicators
            quality_indicators = []
            
            # Check if we have all major indicators
            major_indicators = ["rsi_14", "macd", "vwap", "current_price"]
            available_indicators = sum(1 for indicator in major_indicators if data.get(indicator) is not None)
            quality_indicators.append(available_indicators / len(major_indicators))
            
            # Check for divergence signals (adds confidence)
            if data.get("rsi_divergence", {}).get("bullish_divergence") or data.get("rsi_divergence", {}).get("bearish_divergence"):
                quality_indicators.append(0.2)  # Bonus for divergence detection
            
            # Check multi-timeframe data availability
            mtf_indicators = ["1d_trend", "1wk_trend", "5m_trend"]
            available_mtf = sum(1 for indicator in mtf_indicators if data.get(indicator) is not None)
            quality_indicators.append(available_mtf / len(mtf_indicators))
            
            data_quality = float(np.mean(quality_indicators)) if quality_indicators else 0.5
            
            # Combine consistency and data quality
            confidence = (consistency * 0.6 + data_quality * 0.4)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_scoring_breakdown(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed scoring breakdown for analysis"""
        enhanced_score = self.calculate_enhanced_score(technical_data)
        
        return {
            "enhanced_score": enhanced_score,
            "scoring_weights": self.weights,
            "thresholds": self.thresholds,
            "recommendations": self._get_recommendations(enhanced_score, technical_data)
        }
    
    def _get_recommendations(self, score_data: Dict[str, Any], technical_data: Dict[str, Any]) -> List[str]:
        """Get trading recommendations based on analysis"""
        recommendations = []
        
        signal_strength = score_data.get("signal_strength", "neutral")
        confidence = score_data.get("confidence", 0.5)
        
        if signal_strength in ["strong_buy", "buy"] and confidence > 0.7:
            recommendations.append("Consider long position with tight stop loss")
        
        if signal_strength in ["strong_sell", "sell"] and confidence > 0.7:
            recommendations.append("Consider short position or avoid long positions")
        
        # Divergence recommendations
        if technical_data.get("rsi_divergence", {}).get("bullish_divergence"):
            recommendations.append("Bullish divergence detected - potential reversal signal")
        
        if technical_data.get("rsi_divergence", {}).get("bearish_divergence"):
            recommendations.append("Bearish divergence detected - potential reversal signal")
        
        # Volume recommendations
        current_price = technical_data.get("current_price", 0)
        vwap = technical_data.get("vwap", 0)
        if current_price and vwap and current_price > vwap:
            recommendations.append("Price above VWAP - bullish volume confirmation")
        
        return recommendations

# Singleton instance
enhanced_scoring = EnhancedScoringEngine()
