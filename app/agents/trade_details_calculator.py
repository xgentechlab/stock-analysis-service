"""
Trade Details Calculator for Agent System
Calculates stop loss, targets, and position sizing from agent results
"""
from typing import Dict, Any, Optional
import logging

from app.agents.models import FinalRecommendation, AgentResult
from app.services.stocks import stocks_service

logger = logging.getLogger(__name__)


class AgentTradeDetailsCalculator:
    """Calculate trade details from agent analysis results"""
    
    def __init__(self):
        self.capital_per_trade = 20000  # ₹20,000 per trade
        self.max_risk_percentage = 1.0  # 1% risk per trade
    
    def calculate_trade_details(
        self, recommendation: FinalRecommendation
    ) -> Dict[str, Any]:
        """Calculate comprehensive trade details from agent recommendation"""
        try:
            symbol = recommendation.symbol
            action = recommendation.recommendation.value.lower()
            
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {symbol}")
                return {}
            
            # Extract technical data from Technical Agent
            technical_result = recommendation.agent_breakdown.get('technical')
            technical_metrics = technical_result.metrics if technical_result else {}
            
            # Calculate entry, exit, and position sizing
            entry_details = self._calculate_entry_details(
                symbol, action, current_price, technical_metrics
            )
            
            exit_details = self._calculate_exit_details(
                symbol, action, current_price, technical_metrics
            )
            
            position_sizing = self._calculate_position_sizing(
                current_price, exit_details
            )
            
            risk_metrics = self._calculate_risk_metrics(
                current_price, exit_details, recommendation.confidence
            )
            
            return {
                "entry": entry_details,
                "exit": exit_details,
                "position_sizing": position_sizing,
                "risk_metrics": risk_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade details: {e}")
            return {}
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            price = stocks_service.get_current_price(symbol)
            if price:
                return float(price)
            
            # Fallback: get from OHLCV
            ohlcv = stocks_service.fetch_ohlcv_data(symbol, days=1)
            if ohlcv is not None and not ohlcv.empty:
                return float(ohlcv['Close'].iloc[-1])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return None
    
    def _calculate_entry_details(
        self, symbol: str, action: str, current_price: float,
        technical_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate entry details"""
        if action == 'buy':
            # For BUY: entry at current price or slightly below
            entry_price = current_price
            entry_strategy = "Market order at current price"
            
            # If support level exists, suggest limit order
            support = technical_metrics.get('support_level')
            if support and support < current_price * 0.98:
                entry_price = support
                entry_strategy = f"Limit order near support at ₹{support:.2f}"
        else:
            # For SELL: entry at current price
            entry_price = current_price
            entry_strategy = "Market order at current price"
        
        return {
            "target_entry": entry_price,
            "entry_strategy": entry_strategy,
            "entry_timeframe": "Immediate"
        }
    
    def _calculate_exit_details(
        self, symbol: str, action: str, current_price: float,
        technical_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate exit details (target and stop loss)"""
        if action == 'buy':
            # Target: Use resistance level or calculate from support
            resistance = technical_metrics.get('resistance_level')
            support = technical_metrics.get('support_level', current_price * 0.95)
            
            if resistance and resistance > current_price:
                target_price = resistance
            else:
                # Calculate target based on risk-reward (2:1 minimum)
                risk = current_price - support
                target_price = current_price + (risk * 2)
            
            # Stop loss: Use ATR-based or support level
            atr_stop = technical_metrics.get('atr_stop_loss')
            if atr_stop and atr_stop < current_price:
                stop_loss = atr_stop
            elif support and support < current_price:
                stop_loss = support
            else:
                # Fallback: 5% below current price
                stop_loss = current_price * 0.95
            
            exit_conditions = (
                f"Exit if price hits target ₹{target_price:.2f} or "
                f"stop loss ₹{stop_loss:.2f}"
            )
            
        else:  # SELL
            # For SELL: target is support, stop is resistance
            support = technical_metrics.get('support_level', current_price * 0.95)
            resistance = technical_metrics.get('resistance_level', current_price * 1.05)
            
            target_price = support
            stop_loss = resistance
            
            exit_conditions = (
                f"Exit if price hits target ₹{target_price:.2f} or "
                f"stop loss ₹{stop_loss:.2f}"
            )
        
        return {
            "target_exit": target_price,
            "stop_loss": stop_loss,
            "exit_conditions": exit_conditions
        }
    
    def _calculate_position_sizing(
        self, current_price: float, exit_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate position sizing based on risk"""
        try:
            stop_loss = exit_details.get('stop_loss', current_price * 0.95)
            risk_per_share = abs(current_price - stop_loss)
            
            if risk_per_share == 0:
                risk_per_share = current_price * 0.05  # 5% default
            
            # Maximum risk amount (1% of capital)
            max_risk_amount = self.capital_per_trade * (self.max_risk_percentage / 100)
            
            # Calculate shares based on risk
            recommended_shares = int(max_risk_amount / risk_per_share)
            
            # Don't exceed capital per trade
            max_shares_by_capital = int(self.capital_per_trade / current_price)
            recommended_shares = min(recommended_shares, max_shares_by_capital)
            
            dollar_amount = recommended_shares * current_price
            
            return {
                "recommended_shares": recommended_shares,
                "dollar_amount": dollar_amount,
                "risk_per_share": risk_per_share,
                "max_risk_amount": max_risk_amount
            }
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {
                "recommended_shares": 0,
                "dollar_amount": 0,
                "risk_per_share": 0,
                "max_risk_amount": 0
            }
    
    def _calculate_risk_metrics(
        self, current_price: float, exit_details: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            target_price = exit_details.get('target_exit', current_price * 1.10)
            stop_loss = exit_details.get('stop_loss', current_price * 0.95)
            
            upside = target_price - current_price
            downside = current_price - stop_loss
            
            risk_reward_ratio = upside / downside if downside > 0 else 0
            
            return {
                "upside": upside,
                "downside": downside,
                "risk_reward_ratio": risk_reward_ratio,
                "upside_percentage": (upside / current_price) * 100,
                "downside_percentage": (downside / current_price) * 100,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}


# Singleton instance
agent_trade_details_calculator = AgentTradeDetailsCalculator()

