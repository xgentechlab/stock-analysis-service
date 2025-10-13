"""
Firestore serialization helpers and model utilities
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from app.models.schemas import Signal, Position, Fill, RuntimeConfig, AuditLog

def signal_to_dict(signal: Signal) -> Dict[str, Any]:
    """Convert Signal model to Firestore document"""
    return signal.model_dump()

def dict_to_signal(data: Dict[str, Any]) -> Signal:
    """Convert Firestore document to Signal model"""
    return Signal(**data)

def position_to_dict(position: Position) -> Dict[str, Any]:
    """Convert Position model to Firestore document"""
    return position.model_dump()

def dict_to_position(data: Dict[str, Any]) -> Position:
    """Convert Firestore document to Position model"""
    return Position(**data)

def fill_to_dict(fill: Fill) -> Dict[str, Any]:
    """Convert Fill model to Firestore document"""
    return fill.model_dump()

def dict_to_fill(data: Dict[str, Any]) -> Fill:
    """Convert Firestore document to Fill model"""
    return Fill(**data)

def runtime_config_to_dict(config: RuntimeConfig) -> Dict[str, Any]:
    """Convert RuntimeConfig model to Firestore document"""
    return config.model_dump()

def dict_to_runtime_config(data: Dict[str, Any]) -> RuntimeConfig:
    """Convert Firestore document to RuntimeConfig model"""
    return RuntimeConfig(**data)

def generate_signal_id() -> str:
    """Generate unique signal ID"""
    return str(uuid.uuid4())

def generate_position_id() -> str:
    """Generate unique position ID"""
    return str(uuid.uuid4())

def generate_fill_id() -> str:
    """Generate unique fill ID"""
    return str(uuid.uuid4())

def get_current_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()

def create_meta_block(event_id: Optional[str] = None) -> Dict[str, str]:
    """Create meta block for documents"""
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "created_at": get_current_utc_timestamp()
    }
