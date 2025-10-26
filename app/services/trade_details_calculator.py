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
                "risk_metrics": self._calculate_risk_metrics(current_price, technical_data, risk_data, final_score, confidence),
                "quick_money_impact": self._calculate_quick_money_impact(current_price, risk_data, action)
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
        """Extract analysis summary from job data - FIXED for new 4-stage system"""
        try:
            stages = job_analysis.get('stages', {})
            
            # Use final_score from recommendation (this is the correct calculated value)
            final_score = recommendation.get('final_score', 0.0)
            
            # Get scores from simple_analysis stage (new structure)
            simple_analysis_stage = stages.get('simple_analysis', {})
            simple_analysis_data = simple_analysis_stage.get('data', {})
            
            technical_score = float(simple_analysis_data.get('technical_score', 0.0))
            fundamental_score = float(simple_analysis_data.get('fundamental_score', 0.0))
            risk_level = simple_analysis_data.get('risk_level', 'unknown')
            
            # If not found in simple_analysis, try simple_decision
            if technical_score == 0.0:
                simple_decision_stage = stages.get('simple_decision', {})
                simple_decision_data = simple_decision_stage.get('data', {})
                risk_level = simple_decision_data.get('risk_level', 'unknown')
            
            return {
                "final_score": final_score,
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "risk_level": risk_level
            }
        except Exception as e:
            logger.error(f"Error extracting analysis summary: {e}")
            return {
                "final_score": recommendation.get('final_score', 0.0),
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
        """Extract risk assessment data from job analysis - FIXED for new 4-stage system"""
        try:
            stages = job_analysis.get('stages', {})
            
            # Get risk data from simple_analysis (new structure)
            simple_analysis_stage = stages.get('simple_analysis', {})
            simple_analysis_data = simple_analysis_stage.get('data', {})
            risk_reward_data = simple_analysis_data.get('risk_reward', {})
            
            # Extract risk-reward values
            risk_reward_ratio = float(risk_reward_data.get('risk_reward_ratio', 1.0))
            support_level = float(risk_reward_data.get('support_level', 0.0))
            resistance_level = float(risk_reward_data.get('resistance_level', 0.0))
            downside = float(risk_reward_data.get('downside', 0.0))
            upside = float(risk_reward_data.get('upside', 0.0))
            downside_percentage = float(risk_reward_data.get('downside_percentage', 0.0))
            upside_percentage = float(risk_reward_data.get('upside_percentage', 0.0))
            
            # Get plain english summary for justifications
            plain_english = risk_reward_data.get('plain_english_summary', {})
            ratio_interpretation = risk_reward_data.get('ratio_interpretation', '')
            
            # Get risk level from simple_decision
            risk_level = 'moderate'
            simple_decision_stage = stages.get('simple_decision', {})
            simple_decision_data = simple_decision_stage.get('data', {})
            risk_level = simple_decision_data.get('risk_level', 'moderate')
            
            return {
                "risk_level": risk_level,
                "risk_reward_ratio": risk_reward_ratio,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "downside": downside,
                "upside": upside,
                "downside_percentage": downside_percentage,
                "upside_percentage": upside_percentage,
                "ratio_interpretation": ratio_interpretation,
                "plain_english_summary": plain_english,
                "volatility": 0.0,
                "max_drawdown": 0.0
            }
        except Exception as e:
            logger.error(f"Error extracting risk data: {e}")
            return {
                "risk_level": "moderate",
                "risk_reward_ratio": 1.0,
                "support_level": 0.0,
                "resistance_level": 0.0,
                "downside": 0.0,
                "upside": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0
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
        """Calculate exit details - FIXED to use actual risk_reward data"""
        try:
            rsi = technical_data.get('rsi', 50.0)
            
            # Use actual risk_reward data from analysis
            resistance_level = risk_data.get('resistance_level', 0.0)
            support_level = risk_data.get('support_level', 0.0)
            upside = risk_data.get('upside', 0.0)
            downside = risk_data.get('downside', 0.0)
            risk_reward_ratio = risk_data.get('risk_reward_ratio', 1.0)
            
            if action in ['buy', 'watch']:
                # For buy/watch recommendations: use resistance as target, support as stop loss
                if resistance_level > 0:
                    target_price = resistance_level
                else:
                    target_price = current_price + upside if upside > 0 else current_price * 1.10
                
                if support_level > 0:
                    stop_loss = support_level
                else:
                    stop_loss = current_price - downside if downside > 0 else current_price * 0.95
                
                exit_conditions = f"Exit if RSI > 70 or price hits stop loss at ₹{stop_loss:.2f}"
                
            else:  # sell/avoid
                # For sell recommendations: target support, stop at resistance
                if support_level > 0:
                    target_price = support_level
                else:
                    target_price = current_price - downside if downside > 0 else current_price * 0.92
                
                if resistance_level > 0:
                    stop_loss = resistance_level
                else:
                    stop_loss = current_price + upside if upside > 0 else current_price * 1.05
                
                exit_conditions = f"Exit if RSI < 30 or price hits stop loss at ₹{stop_loss:.2f}"
            
            return {
                "target_price": target_price,
                "stop_loss": stop_loss,
                "exit_conditions": exit_conditions,
                "timeframe": "2-4 weeks",
                "risk_reward_ratio": risk_reward_ratio,
                "justification": risk_data.get('ratio_interpretation', ''),
                "plain_english": risk_data.get('plain_english_summary', {})
            }
        except Exception as e:
            logger.error(f"Error calculating exit details: {e}")
            return {
                "target_price": current_price,
                "stop_loss": current_price * 0.95,
                "exit_conditions": "Exit based on market conditions",
                "timeframe": "2-4 weeks",
                "risk_reward_ratio": risk_data.get('risk_reward_ratio', 1.0),
                "justification": "",
                "plain_english": {}
            }
    
    def _calculate_position_sizing(self, current_price: float, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position sizing based on capital and risk - FIXED for expensive stocks"""
        try:
            # Calculate shares based on capital
            recommended_shares = int(self.capital_per_trade / current_price)
            
            # If the stock is too expensive, provide fractional info
            if recommended_shares == 0:
                fractional_shares = self.capital_per_trade / current_price
                portfolio_percentage = 100.0
                actual_dollar_amount = self.capital_per_trade  # Use full capital
            else:
                fractional_shares = recommended_shares
                portfolio_percentage = 100.0
                actual_dollar_amount = recommended_shares * current_price
            
            # Risk per trade
            risk_per_trade = self.max_risk_percentage
            
            return {
                "recommended_shares": recommended_shares,
                "fractional_shares": round(fractional_shares, 4) if recommended_shares == 0 else None,
                "portfolio_percentage": portfolio_percentage,
                "dollar_amount": actual_dollar_amount,
                "risk_per_trade": risk_per_trade,
                "note": f"Position size based on ₹{self.capital_per_trade:,.0f} capital" if recommended_shares == 0 else None
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
        """Calculate risk metrics - FIXED to use actual data"""
        try:
            # Use actual risk_reward_ratio from analysis
            risk_reward_ratio = float(risk_data.get('risk_reward_ratio', 1.0))
            downside = float(risk_data.get('downside', 0.0))
            upside = float(risk_data.get('upside', 0.0))
            
            # Calculate max risk amount based on downside
            if downside > 0:
                max_risk_amount = (downside / current_price) * self.capital_per_trade
            else:
                max_risk_amount = self.capital_per_trade * (self.max_risk_percentage / 100)
            
            # ATR stop distance
            atr = technical_data.get('atr', current_price * 0.02)
            atr_stop_distance = (atr / current_price) * 100  # As percentage
            
            return {
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "potential_upside": round(upside, 2) if upside > 0 else 0.0,
                "potential_downside": round(downside, 2) if downside > 0 else 0.0,
                "max_risk_amount": round(max_risk_amount, 2),
                "confidence_level": round(confidence, 2),
                "atr_stop_distance": round(atr_stop_distance, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "risk_reward_ratio": 1.0,
                "potential_upside": 0.0,
                "potential_downside": 0.0,
                "max_risk_amount": 200.0,
                "confidence_level": 0.5,
                "atr_stop_distance": 2.0
            }
    
    def _calculate_quick_money_impact(self, current_price: float, risk_data: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Reuse existing money impact data from analysis"""
        try:
            # Try to extract from risk_reward.plain_english_summary
            # This already has the formatted data we need
            stages = risk_data.get('_stages', {}) if isinstance(risk_data, dict) else {}
            
            # For now, extract what we have in risk_data
            upside = float(risk_data.get('upside', 0.0))
            downside = float(risk_data.get('downside', 0.0))
            risk_reward_ratio = float(risk_data.get('risk_reward_ratio', 1.0))
            
            if upside > 0 and downside > 0:
                # Calculate for ₹10,000
                investment_amount = 10000.0
                shares = investment_amount / current_price
                potential_gain = shares * upside
                potential_loss = shares * downside
                
                # Get percentages from risk_data
                upside_pct = float(risk_data.get('upside_percentage', 0.0))
                downside_pct = float(risk_data.get('downside_percentage', 0.0))
                
                net_risk_reward = f"Risk ₹{potential_loss:.0f} to make ₹{potential_gain:.0f}"
                
                return {
                    "investment_amount": investment_amount,
                    "shares": round(shares, 4),
                    "potential_gain": round(potential_gain, 2),
                    "potential_loss": round(potential_loss, 2),
                    "gain_percentage": round(upside_pct, 2),
                    "loss_percentage": round(downside_pct, 2),
                    "net_risk_reward": net_risk_reward,
                    "risk_reward_ratio": round(risk_reward_ratio, 2),
                    "action": action,
                    "summary": f"For ₹10,000: {net_risk_reward}"
                }
            else:
                # Fallback if data not available
                return {
                    "investment_amount": 10000.0,
                    "shares": round(10000.0 / current_price, 4),
                    "potential_gain": 0.0,
                    "potential_loss": 0.0,
                    "gain_percentage": 0.0,
                    "loss_percentage": 0.0,
                    "net_risk_reward": "Data not available",
                    "risk_reward_ratio": 1.0,
                    "action": action,
                    "summary": "Unable to calculate"
                }
        except Exception as e:
            logger.error(f"Error calculating quick money impact: {e}")
            return {
                "investment_amount": 10000.0,
                "shares": 0.0,
                "potential_gain": 0.0,
                "potential_loss": 0.0,
                "gain_percentage": 0.0,
                "loss_percentage": 0.0,
                "net_risk_reward": "Unable to calculate",
                "risk_reward_ratio": 1.0,
                "action": action,
                "summary": "Unable to calculate money impact"
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
