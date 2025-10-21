"""
Trade Details Calculator Service
Calculates trade details from analysis data for recommendations
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.services.stocks import stocks_service

logger = logging.getLogger(__name__)

class TradeDetailsCalculator:
    """Service for calculating trade details from analysis data"""
    
    def __init__(self):
        self.capital_per_trade = 20000  # Hardcoded ₹20,000 per trade
        self.max_risk_percentage = 1.0  # 1% risk per trade
    
    def calculate_trade_details(self, recommendation: Dict[str, Any], job_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive trade details from recommendation and analysis data"""
        try:
            symbol = recommendation.get('symbol')
            action = recommendation.get('action', '').lower()
            final_score = recommendation.get('final_score', 0.0)
            confidence = recommendation.get('confidence', 0.0)
            
            logger.info(f"Calculating trade details for {symbol} ({action})")
            
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not fetch current price for {symbol}")
                return self._create_error_response("Unable to fetch current price")
            
            # Extract analysis data
            analysis_summary = self._extract_analysis_summary(job_analysis, recommendation)
            technical_data = self._extract_technical_data(job_analysis)
            risk_data = self._extract_risk_data(job_analysis)
            
            # Calculate trade details
            trade_details = {
                "entry": self._calculate_entry_details(symbol, action, current_price, technical_data),
                "exit": self._calculate_exit_details(symbol, action, current_price, technical_data, risk_data),
                "position_sizing": self._calculate_position_sizing(current_price, risk_data),
                "risk_metrics": self._calculate_risk_metrics(current_price, technical_data, risk_data, final_score, confidence)
            }
            
            return {
                "recommendation_id": recommendation.get('id'),
                "symbol": symbol,
                "action": action,
                "current_price": current_price,
                "trade_details": trade_details,
                "analysis_summary": analysis_summary
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade details: {e}")
            return self._create_error_response(str(e))
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # Use existing stocks service to get current price
            price_data = stocks_service.get_current_price(symbol)
            if price_data and isinstance(price_data, (int, float)):
                return float(price_data)
            
            # Fallback: try to get from OHLCV data
            ohlcv_data = stocks_service.fetch_ohlcv_data(symbol, days=1)
            if ohlcv_data is not None and not ohlcv_data.empty:
                return float(ohlcv_data['Close'].iloc[-1])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def _extract_analysis_summary(self, job_analysis: Dict[str, Any], recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract analysis summary from job data"""
        try:
            stages = job_analysis.get('stages', {})
            
            # Use final_score from recommendation (this is the correct calculated value)
            final_score = recommendation.get('final_score', 0.0)
            
            # Get technical scoring from correct path (stages have 'data' key)
            technical_stage = stages.get('technical_and_combined_scoring', {})
            technical_data = technical_stage.get('data', {})
            technical_scoring = technical_data.get('combined_scoring', {})
            technical_score = technical_scoring.get('technical_score', 0.0)
            # Get fundamental scoring from correct path
            data_stage = stages.get('data_collection_and_analysis', {})
            data_collection_data = data_stage.get('data', {})
            fundamental_analysis = data_collection_data.get('fundamental_analysis', {})
            fundamental_score_data = fundamental_analysis.get('fundamental_score', {})
            fundamental_score = fundamental_score_data.get('final_score', 0.0)
            
            # Get risk assessment
            risk_stage = stages.get('risk_assessment', {})
            risk_data = risk_stage.get('data', {})
            risk_level = risk_data.get('risk_level', 'unknown')
            
            return {
                "final_score": final_score,
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "risk_level": risk_level
            }
        except Exception as e:
            logger.error(f"Error extracting analysis summary: {e}")
            return {
                "final_score": 0.0,
                "technical_score": 0.0,
                "fundamental_score": 0.0,
                "risk_level": "unknown"
            }
    
    def _extract_technical_data(self, job_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical indicators from job analysis"""
        try:
            stages = job_analysis.get('stages', {})
            data_stage = stages.get('data_collection_and_analysis', {})
            data_collection = data_stage.get('data', {})
            technical_analysis = data_collection.get('technical_analysis', {})
            technical_indicators = technical_analysis.get('technical_indicators', {})
            basic_indicators = technical_indicators.get('basic_indicators', {})
            
            return {
                "rsi": basic_indicators.get('rsi_14', 50.0),
                "sma_20": basic_indicators.get('sma_20', 0.0),
                "sma_50": basic_indicators.get('sma_50', 0.0),
                "atr": basic_indicators.get('atr_14', 0.0),
                "current_price": data_collection.get('current_price', 0.0),
                "volume": technical_indicators.get('volume_indicators', {}).get('obv', 0),
                "price_change_1d": technical_analysis.get('price_change_1d', 0.0),
                "price_change_5d": technical_analysis.get('price_change_5d', 0.0)
            }
        except Exception as e:
            logger.error(f"Error extracting technical data: {e}")
            return {}
    
    def _extract_risk_data(self, job_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment data from job analysis"""
        try:
            stages = job_analysis.get('stages', {})
            risk_stage = stages.get('risk_assessment', {})
            risk_data = risk_stage.get('data', {})
            
            # Get risk-reward ratio from forensic analysis or risk assessment
            forensic_stage = stages.get('forensic_analysis', {})
            forensic_data = forensic_stage.get('data', {})
            risk_reward_calc = forensic_data.get('risk_reward_calculation', {})
            risk_reward_ratio = risk_reward_calc.get('risk_reward_ratio', 1.0)
            
            return {
                "risk_level": risk_data.get('risk_level', 'moderate'),
                "volatility": risk_data.get('volatility', 0.0),
                "max_drawdown": risk_data.get('max_drawdown', 0.0),
                "risk_reward_ratio": risk_reward_ratio
            }
        except Exception as e:
            logger.error(f"Error extracting risk data: {e}")
            return {
                "risk_level": "moderate",
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "risk_reward_ratio": 1.0
            }
    
    def _calculate_entry_details(self, symbol: str, action: str, current_price: float, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate entry details"""
        try:
            rsi = technical_data.get('rsi', 50.0)
            sma_20 = technical_data.get('sma_20', current_price)
            sma_50 = technical_data.get('sma_50', current_price)
            
            if action == 'buy':
                # For buy recommendations
                target_entry = current_price * 1.002  # 0.2% above current price
                entry_conditions = f"Buy if price breaks above ₹{target_entry:.2f} with volume confirmation"
                
                # Add RSI condition
                if rsi < 30:
                    entry_conditions += " (Oversold - good entry opportunity)"
                elif rsi > 70:
                    entry_conditions += " (Overbought - wait for pullback)"
                    
            else:  # sell
                # For sell recommendations
                target_entry = current_price * 0.998  # 0.2% below current price
                entry_conditions = f"Sell if price breaks below ₹{target_entry:.2f} with volume confirmation"
                
                # Add RSI condition
                if rsi > 70:
                    entry_conditions += " (Overbought - good exit opportunity)"
                elif rsi < 30:
                    entry_conditions += " (Oversold - wait for bounce)"
            
            return {
                "current_price": current_price,
                "target_entry": target_entry,
                "entry_conditions": entry_conditions,
                "entry_timeframe": "2-3 trading days"
            }
        except Exception as e:
            logger.error(f"Error calculating entry details: {e}")
            return {
                "current_price": current_price,
                "target_entry": current_price,
                "entry_conditions": "Enter at current market price",
                "entry_timeframe": "Immediate"
            }
    
    def _calculate_exit_details(self, symbol: str, action: str, current_price: float, technical_data: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate exit details"""
        try:
            rsi = technical_data.get('rsi', 50.0)
            atr = technical_data.get('atr', current_price * 0.02)  # Default 2% ATR
            risk_reward_ratio = risk_data.get('risk_reward_ratio', 1.5)
            
            if action == 'buy':
                # For buy recommendations
                # Target: 5-15% upside based on confidence
                target_percentage = 0.08  # 8% default
                target_price = current_price * (1 + target_percentage)
                
                # Stop loss: 2-5% downside or ATR-based
                stop_percentage = max(0.03, min(0.05, atr / current_price))  # 3-5% or ATR
                stop_loss = current_price * (1 - stop_percentage)
                
                exit_conditions = f"Exit if RSI > 70 or price hits stop loss at ₹{stop_loss:.2f}"
                
            else:  # sell
                # For sell recommendations
                # Target: 5-15% downside
                target_percentage = 0.08  # 8% default
                target_price = current_price * (1 - target_percentage)
                
                # Stop loss: 2-5% upside
                stop_percentage = max(0.03, min(0.05, atr / current_price))
                stop_loss = current_price * (1 + stop_percentage)
                
                exit_conditions = f"Exit if RSI < 30 or price hits stop loss at ₹{stop_loss:.2f}"
            
            return {
                "target_price": target_price,
                "stop_loss": stop_loss,
                "exit_conditions": exit_conditions,
                "timeframe": "2-4 weeks"
            }
        except Exception as e:
            logger.error(f"Error calculating exit details: {e}")
            return {
                "target_price": current_price,
                "stop_loss": current_price * 0.95,
                "exit_conditions": "Exit based on market conditions",
                "timeframe": "2-4 weeks"
            }
    
    def _calculate_position_sizing(self, current_price: float, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position sizing based on capital and risk"""
        try:
            # Calculate shares based on capital
            recommended_shares = int(self.capital_per_trade / current_price)
            
            # Calculate actual dollar amount
            actual_dollar_amount = recommended_shares * current_price
            
            # Calculate portfolio percentage (100% since it's per trade)
            portfolio_percentage = 100.0
            
            # Risk per trade
            risk_per_trade = self.max_risk_percentage
            
            return {
                "recommended_shares": recommended_shares,
                "portfolio_percentage": portfolio_percentage,
                "dollar_amount": actual_dollar_amount,
                "risk_per_trade": risk_per_trade
            }
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {
                "recommended_shares": 0,
                "portfolio_percentage": 0.0,
                "dollar_amount": 0.0,
                "risk_per_trade": 0.0
            }
    
    def _calculate_risk_metrics(self, current_price: float, technical_data: Dict[str, Any], risk_data: Dict[str, Any], final_score: float, confidence: float) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            atr = technical_data.get('atr', current_price * 0.02)
            risk_reward_ratio = risk_data.get('risk_reward_ratio', 1.5)
            
            # Calculate max risk amount (1% of capital)
            max_risk_amount = self.capital_per_trade * (self.max_risk_percentage / 100)
            
            # ATR stop distance
            atr_stop_distance = (atr / current_price) * 100  # As percentage
            
            return {
                "risk_reward_ratio": risk_reward_ratio,
                "max_risk_amount": max_risk_amount,
                "confidence_level": confidence,
                "atr_stop_distance": atr_stop_distance
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "risk_reward_ratio": 1.0,
                "max_risk_amount": 200.0,
                "confidence_level": 0.5,
                "atr_stop_distance": 2.0
            }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "error": error_message,
            "recommendation_id": None,
            "symbol": None,
            "action": None,
            "current_price": None,
            "trade_details": None,
            "analysis_summary": None
        }

# Create singleton instance
trade_details_calculator = TradeDetailsCalculator()
