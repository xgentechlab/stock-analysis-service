from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum

class TechnicalSnapshot(BaseModel):
    close: float
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    rsi14: Optional[float] = None
    atr14: Optional[float] = None
    vol20: Optional[int] = None
    vol_today: Optional[int] = None
    pct_1d: Optional[float] = None
    pct_5d: Optional[float] = None

class FundamentalSnapshot(BaseModel):
    pe: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    eps_ttm: Optional[float] = None
    market_cap_cr: Optional[float] = None

class Verdict(BaseModel):
    action: str  # buy|watch|avoid
    confidence: float = Field(ge=0.0, le=1.0)
    explain: str

class Signal(BaseModel):
    signal_id: str
    symbol: str
    venue: str = "NSE"
    created_at: str
    status: str = "open"  # open | placed | dismissed | expired
    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    fundamentals: FundamentalSnapshot
    technical: TechnicalSnapshot
    meta: Dict[str, Any] = {}
    expiry: str

class Position(BaseModel):
    position_id: str
    signal_id: str
    symbol: str
    venue: str = "NSE"
    qty: int
    avg_price: float
    entered_at: str
    status: str = "open"  # open | closed
    exit_rules: Dict[str, Any] = {}
    latest_price: Optional[float] = None
    unrealized_minor: Optional[int] = None

class Fill(BaseModel):
    fill_id: str
    position_id: str
    side: str

# Job and Stage Models for Async Processing
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class AnalysisType(str, Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"

class JobStage(BaseModel):
    stage_name: str
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None

class Job(BaseModel):
    job_id: str
    symbol: str
    analysis_type: AnalysisType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    estimated_time: Optional[int] = None  # seconds
    actual_time: Optional[float] = None  # seconds
    error: Optional[str] = None
    stages: Dict[str, JobStage] = Field(default_factory=dict)
    priority: Literal["low", "normal", "high"] = "normal"
    created_by: Optional[str] = None

class JobCreateRequest(BaseModel):
    symbol: str
    analysis_type: AnalysisType = AnalysisType.ENHANCED
    priority: Literal["low", "normal", "high"] = "normal"

class JobStatusResponse(BaseModel):
    job_id: str
    symbol: str
    analysis_type: AnalysisType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    actual_time: Optional[float] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    priority: Literal["low", "normal", "high"] = "normal"

class StageDataResponse(BaseModel):
    stage_name: str
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    dependencies: List[str] = Field(default_factory=list)

class StageMapping(BaseModel):
    stage_name: str
    display_name: str
    description: str
    estimated_duration: int  # seconds
    dependencies: List[str] = Field(default_factory=list)
    order: int

class RuntimeConfig(BaseModel):
    paper_mode: bool = True
    kill_switch: bool = False
    max_order_notional_minor: int = 2000000

class AuditLog(BaseModel):
    id: str
    action: str
    details: Dict[str, Any]
    timestamp: str
    source: str

# Request/Response DTOs
class PlaceOrderRequest(BaseModel):
    qty: int = Field(gt=0)
    stop: Optional[float] = None
    target: Optional[float] = None

class ApiResponse(BaseModel):
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None

class SignalsListResponse(BaseModel):
    signals: List[Signal]
    total: int
    cursor: Optional[str] = None

class PositionsListResponse(BaseModel):
    positions: List[Position]
    total: int

class SelectionSummary(BaseModel):
    count_in: int
    top_symbols: List[str]
    signals_created: int
    run_duration_seconds: float

class TrackerSummary(BaseModel):
    closed_count: int
    updated_count: int
    positions_checked: int
    run_duration_seconds: float
