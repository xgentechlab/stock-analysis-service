"""
Order service for handling trade execution
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

from app.db.firestore_client import firestore_client
from app.models.firestore_models import generate_position_id, generate_fill_id

logger = logging.getLogger(__name__)

class OrderService:
    def __init__(self):
        pass
    
    async def place_order(self, signal_id: str, signal_data: Dict[str, Any], 
                         qty: int, stop: Optional[float] = None, 
                         target: Optional[float] = None, paper_mode: bool = True) -> Dict[str, Any]:
        """
        Place an order for a signal
        Creates position and fill records, updates signal status
        """
        try:
            symbol = signal_data["symbol"]
            current_price = signal_data["technical"]["close"]
            
            # Create position
            position_id = generate_position_id()
            
            # Set up exit rules
            exit_rules = {}
            if stop:
                exit_rules["stop"] = stop
            if target:
                exit_rules["target"] = target
            
            position_data = {
                "position_id": position_id,
                "signal_id": f"signals/{signal_id}",
                "symbol": symbol,
                "venue": "NSE",
                "qty": qty,
                "avg_price": current_price,
                "entered_at": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "exit_rules": exit_rules,
                "latest_price": current_price,
                "unrealized_minor": 0
            }
            
            # Create position in Firestore
            created_position_id = firestore_client.create_position(position_data)
            
            # Create fill record
            fill_data = {
                "position_id": f"positions/{created_position_id}",
                "side": "buy",
                "qty": qty,
                "price": current_price,
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "paper" if paper_mode else "broker"
            }
            
            fill_id = firestore_client.create_fill(fill_data)
            
            # Update signal status to placed
            firestore_client.update_signal_status(signal_id, "placed")
            
            # Create audit log
            firestore_client.create_audit_log(
                action="order_placed",
                details={
                    "signal_id": signal_id,
                    "position_id": created_position_id,
                    "fill_id": fill_id,
                    "symbol": symbol,
                    "qty": qty,
                    "price": current_price,
                    "paper_mode": paper_mode,
                    "exit_rules": exit_rules
                },
                source="order_service"
            )
            
            logger.info(f"Order placed: {symbol} x{qty} @ ₹{current_price} "
                       f"(Position: {created_position_id}, Paper: {paper_mode})")
            
            # Return position data
            return {
                "position_id": created_position_id,
                "symbol": symbol,
                "qty": qty,
                "avg_price": current_price,
                "status": "open",
                "paper_mode": paper_mode,
                "exit_rules": exit_rules
            }
            
        except Exception as e:
            logger.error(f"Error placing order for signal {signal_id}: {e}")
            raise

# Singleton instance
order_service = OrderService()
