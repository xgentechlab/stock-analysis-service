"""
Position tracker service for monitoring open positions and applying exit rules
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

from app.services.stocks import stocks_service
from app.db.firestore_client import firestore_client

logger = logging.getLogger(__name__)

class PositionTracker:
    def __init__(self):
        pass
    
    def run_position_tracking(self) -> Dict[str, Any]:
        """
        Run position tracking cycle
        - Fetch all open positions
        - Update latest prices and unrealized P&L
        - Check exit rules and close positions if needed
        """
        start_time = time.time()
        
        logger.info("Starting position tracking cycle")
        
        try:
            # Get all open positions
            open_positions = firestore_client.list_positions(status="open")
            logger.info(f"Found {len(open_positions)} open positions")
            
            if not open_positions:
                return {
                    "closed_count": 0,
                    "updated_count": 0,
                    "positions_checked": 0,
                    "run_duration_seconds": round(time.time() - start_time, 2)
                }
            
            closed_count = 0
            updated_count = 0
            
            for position in open_positions:
                try:
                    result = self._process_position(position)
                    if result == "closed":
                        closed_count += 1
                    elif result == "updated":
                        updated_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing position {position.get('position_id')}: {e}")
                    continue
            
            duration = time.time() - start_time
            
            summary = {
                "closed_count": closed_count,
                "updated_count": updated_count,
                "positions_checked": len(open_positions),
                "run_duration_seconds": round(duration, 2)
            }
            
            logger.info(f"Position tracking completed in {duration:.2f}s. "
                       f"Updated: {updated_count}, Closed: {closed_count}")
            
            # Create audit log
            firestore_client.create_audit_log(
                action="position_tracking_completed",
                details=summary,
                source="position_tracker"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in position tracking: {e}")
            
            # Create error audit log
            firestore_client.create_audit_log(
                action="position_tracking_failed",
                details={"error": str(e)},
                source="position_tracker"
            )
            
            raise
    
    def _process_position(self, position: Dict[str, Any]) -> str:
        """
        Process a single position
        Returns: "closed", "updated", or "unchanged"
        """
        position_id = position['position_id']
        symbol = position['symbol']
        
        try:
            # Get current price
            current_price = stocks_service.get_current_price(symbol)
            if current_price is None:
                logger.warning(f"Could not fetch current price for {symbol}")
                return "unchanged"
            
            # Calculate unrealized P&L
            avg_price = position['avg_price']
            qty = position['qty']
            
            # Assuming long positions for now
            unrealized_minor = int((current_price - avg_price) * qty * 100)  # Convert to paise
            
            # Check if position should be closed based on exit rules
            should_close, close_reason = self._check_exit_rules(
                position, current_price, unrealized_minor
            )
            
            if should_close:
                # Close the position
                self._close_position(position_id, current_price, close_reason)
                return "closed"
            else:
                # Update position with latest price and P&L
                updates = {
                    "latest_price": current_price,
                    "unrealized_minor": unrealized_minor,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
                firestore_client.update_position(position_id, updates)
                return "updated"
                
        except Exception as e:
            logger.error(f"Error processing position {position_id}: {e}")
            return "unchanged"
    
    def _check_exit_rules(self, position: Dict[str, Any], current_price: float, 
                         unrealized_minor: int) -> tuple[bool, Optional[str]]:
        """
        Check if position should be closed based on exit rules
        Returns (should_close: bool, reason: str)
        """
        exit_rules = position.get('exit_rules', {})
        avg_price = position['avg_price']
        
        # Stop loss check
        stop_price = exit_rules.get('stop')
        if stop_price and current_price <= stop_price:
            return True, f"stop_loss_hit:{stop_price}"
        
        # Target profit check
        target_price = exit_rules.get('target')
        if target_price and current_price >= target_price:
            return True, f"target_hit:{target_price}"
        
        # Default stop loss (if not specified) - 5% loss
        if not stop_price:
            loss_threshold = avg_price * 0.95  # 5% stop loss
            if current_price <= loss_threshold:
                return True, f"default_stop_loss:{loss_threshold:.2f}"
        
        # Time-based exit - close if position older than 5 days
        entered_at = datetime.fromisoformat(position['entered_at'].replace('Z', '+00:00'))
        days_held = (datetime.now(timezone.utc) - entered_at).days
        
        if days_held >= 5:
            return True, f"time_exit:{days_held}_days"
        
        # Trailing stop loss - 3% from recent high
        # This would require tracking the highest price seen
        # For now, implement a simple version
        
        return False, None
    
    def _close_position(self, position_id: str, exit_price: float, close_reason: str):
        """
        Close a position by creating a sell fill and updating position status
        """
        try:
            # Get position details
            position = firestore_client.get_position(position_id)
            if not position:
                logger.error(f"Position {position_id} not found for closing")
                return
            
            # Create exit fill
            fill_data = {
                "position_id": f"positions/{position_id}",
                "side": "sell",
                "qty": position['qty'],
                "price": exit_price,
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "paper",  # Paper trading for now
                "exit_reason": close_reason
            }
            
            fill_id = firestore_client.create_fill(fill_data)
            
            # Calculate realized P&L
            avg_price = position['avg_price']
            qty = position['qty']
            realized_minor = int((exit_price - avg_price) * qty * 100)  # Convert to paise
            
            # Update position to closed
            updates = {
                "status": "closed",
                "latest_price": exit_price,
                "exit_price": exit_price,
                "exit_reason": close_reason,
                "realized_minor": realized_minor,
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "exit_fill_id": fill_id
            }
            
            firestore_client.update_position(position_id, updates)
            
            logger.info(f"Closed position {position_id} at ₹{exit_price} "
                       f"(P&L: ₹{realized_minor/100:.2f}, Reason: {close_reason})")
            
            # Create audit log for position closure
            firestore_client.create_audit_log(
                action="position_closed",
                details={
                    "position_id": position_id,
                    "symbol": position['symbol'],
                    "exit_price": exit_price,
                    "realized_pnl_minor": realized_minor,
                    "close_reason": close_reason
                },
                source="position_tracker"
            )
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            raise
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        try:
            open_positions = firestore_client.list_positions(status="open")
            closed_positions = firestore_client.list_positions(status="closed")
            
            # Calculate summary stats
            total_open_value = 0
            total_unrealized = 0
            
            for pos in open_positions:
                if pos.get('avg_price') and pos.get('qty'):
                    total_open_value += pos['avg_price'] * pos['qty']
                
                if pos.get('unrealized_minor'):
                    total_unrealized += pos['unrealized_minor']
            
            total_realized = 0
            for pos in closed_positions:
                if pos.get('realized_minor'):
                    total_realized += pos['realized_minor']
            
            return {
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "total_open_value": total_open_value,
                "total_unrealized_minor": total_unrealized,
                "total_realized_minor": total_realized,
                "net_pnl_minor": total_unrealized + total_realized
            }
            
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {}
    
    def force_close_position(self, position_id: str, reason: str = "manual_close") -> bool:
        """Manually force close a position"""
        try:
            position = firestore_client.get_position(position_id)
            if not position:
                logger.error(f"Position {position_id} not found")
                return False
            
            if position['status'] != 'open':
                logger.warning(f"Position {position_id} is not open")
                return False
            
            # Get current price
            current_price = stocks_service.get_current_price(position['symbol'])
            if current_price is None:
                logger.error(f"Could not fetch current price for {position['symbol']}")
                return False
            
            # Close the position
            self._close_position(position_id, current_price, reason)
            
            logger.info(f"Manually closed position {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error force closing position {position_id}: {e}")
            return False

# Singleton instance
position_tracker = PositionTracker()
