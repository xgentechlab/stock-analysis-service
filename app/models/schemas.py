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

class MultiTimeframeData(BaseModel):
    """Multi-timeframe OHLCV data for technical analysis"""
    timeframe: str  # 1m, 5m, 15m, 1d, 1wk
    data: List[Dict[str, Any]]  # OHLCV data points
    last_updated: datetime
    data_points: int

class MultiTimeframeAnalysis(BaseModel):
    """Comprehensive multi-timeframe analysis for a stock"""
    symbol: str
    analysis_id: str
    created_at: datetime
    updated_at: datetime
    
    # Technical indicators across timeframes
    technical_indicators: Dict[str, Any] = Field(default_factory=dict)
    
    # Trend analysis
    trend_alignment: Dict[str, str] = Field(default_factory=dict)  # timeframe -> trend
    momentum_scores: Dict[str, float] = Field(default_factory=dict)  # timeframe -> score
    
    # Divergence signals
    divergence_signals: Dict[str, Any] = Field(default_factory=dict)
    
    # Volume analysis
    volume_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Overall multi-timeframe score
    mtf_score: Optional[float] = None
    mtf_confidence: Optional[float] = None
    mtf_strength: Optional[str] = None  # strong, moderate, weak
    
    # Metadata
    analysis_version: str = "1.0"
    data_quality: str = "good"  # good, fair, poor

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

# User Management Models
class RecommendationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class RecommendationPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class RecommendationAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class DecisionType(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ADD_TO_WATCHLIST = "add_to_watchlist"

class PortfolioStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"

class SuggestionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"

# Trade Details Model
class TradeDetails(BaseModel):
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    quantity: Optional[int] = None
    position_size: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    confidence: Optional[float] = None

# Recommendation Model
class Recommendation(BaseModel):
    id: str
    user_id: str
    symbol: str
    action: RecommendationAction
    reason: str
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    created_at: str
    status: RecommendationStatus = RecommendationStatus.PENDING
    final_score: Optional[float] = None
    source_job_id: Optional[str] = None
    trade_details: Optional[TradeDetails] = None

class RecommendationCreateRequest(BaseModel):
    user_id: str
    symbol: str
    action: RecommendationAction
    reason: str
    priority: RecommendationPriority = RecommendationPriority.MEDIUM

class RecommendationUpdateRequest(BaseModel):
    status: Optional[RecommendationStatus] = None
    reason: Optional[str] = None
    priority: Optional[RecommendationPriority] = None

# Watchlist Model
class WatchlistItem(BaseModel):
    user_id: str
    symbol: str
    added_at: str

class WatchlistAddRequest(BaseModel):
    user_id: str
    symbol: str

# Portfolio Model (Enhanced version of existing Position)
class PortfolioItem(BaseModel):
    id: str
    user_id: str
    symbol: str
    quantity: int
    avg_price: float  # Price per share at entry
    entry_date: str
    current_price: Optional[float] = None  # Current price per share
    current_value: Optional[float] = None  # Total current value (current_price * quantity)
    invested_amount: Optional[float] = None  # Total amount invested (avg_price * quantity)
    pnl: Optional[float] = None  # Profit/Loss
    status: PortfolioStatus = PortfolioStatus.ACTIVE

class PortfolioCreateRequest(BaseModel):
    user_id: str
    symbol: str
    quantity: int
    avg_price: float
    entry_date: Optional[str] = None

class PortfolioUpdateRequest(BaseModel):
    quantity: Optional[int] = None
    current_value: Optional[float] = None
    pnl: Optional[float] = None
    status: Optional[PortfolioStatus] = None

# User Decision Model
class UserDecisionRecord(BaseModel):
    id: str
    recommendation_id: str
    decision: DecisionType
    decided_at: str
    user_id: Optional[str] = None
    symbol: Optional[str] = None
    action_taken: Optional[str] = None

class UserDecisionCreateRequest(BaseModel):
    recommendation_id: str
    decision: DecisionType
    user_id: Optional[str] = None
    symbol: Optional[str] = None
    action_taken: Optional[str] = None

# Portfolio Suggestion Model
class PortfolioSuggestion(BaseModel):
    id: str
    user_id: str
    remove_symbol: Optional[str] = None
    add_symbol: Optional[str] = None
    reason: str
    created_at: str
    status: SuggestionStatus = SuggestionStatus.PENDING

class PortfolioSuggestionCreateRequest(BaseModel):
    user_id: str
    remove_symbol: Optional[str] = None
    add_symbol: Optional[str] = None
    reason: str

class PortfolioSuggestionUpdateRequest(BaseModel):
    status: SuggestionStatus

# Response Models
class RecommendationsListResponse(BaseModel):
    recommendations: List[Recommendation]
    total: int
    cursor: Optional[str] = None

class WatchlistListResponse(BaseModel):
    watchlist: List[WatchlistItem]
    total: int

class PortfolioListResponse(BaseModel):
    portfolio: List[PortfolioItem]
    total: int

class UserDecisionsListResponse(BaseModel):
    decisions: List[UserDecisionRecord]
    total: int

class PortfolioSuggestionsListResponse(BaseModel):
    suggestions: List[PortfolioSuggestion]
    total: int

# News Intelligence Models
class NewsworthyStock(BaseModel):
    symbol: str
    name: Optional[str] = None
    mentions: int
    sentiment: float = Field(ge=-1.0, le=1.0)
    sources: List[str] = []
    excerpt: Optional[str] = None

class NewsIntelligenceResponse(BaseModel):
    stocks: List[NewsworthyStock]
    total: int

# Stock Model for Nifty 500 stocks
class Stock(BaseModel):
    id: str
    company_name: str
    symbol: str
    industry: str
    series: Optional[str] = None
    isin_code: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: bool = True

class StockCreateRequest(BaseModel):
    company_name: str
    symbol: str
    industry: str
    series: Optional[str] = None
    isin_code: Optional[str] = None

class StockUpdateRequest(BaseModel):
    company_name: Optional[str] = None
    symbol: Optional[str] = None
    industry: Optional[str] = None
    series: Optional[str] = None
    isin_code: Optional[str] = None
    is_active: Optional[bool] = None

class StockListResponse(BaseModel):
    stocks: List[Stock]
    total: int
    page: int = 1
    limit: int = 50

# Hot Stocks Models
class HotStockSelection(BaseModel):
    """Individual hot stock selection with analysis data"""
    symbol: str
    rank: int
    enhanced_technical_score: Optional[float] = None
    enhanced_fundamental_score: Optional[float] = None
    enhanced_combined_score: Optional[float] = None
    basic_composite_score: Optional[float] = None
    basic_fundamental_score: Optional[float] = None
    momentum_pct_5d: float
    volume_spike_ratio: float
    institutional_activity: bool = False
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    market_cap_cr: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    analysis_id: str  # Reference to detailed analysis
    multi_timeframe_analysis_id: Optional[str] = None

class HotStocksRunMetadata(BaseModel):
    """Metadata for a hot stocks run"""
    run_id: str
    run_timestamp: datetime
    universe_size: int
    total_processed: int
    total_filtered: int
    total_selected: int
    processing_time_seconds: float
    filters_applied: Dict[str, Any]
    selection_criteria: Dict[str, Any]
    data_quality: str = "good"
    stage_1_2_integrated: bool = True
    data_fetch_optimized: bool = True
    api_version: str = "1.0"

class HotStocksRun(BaseModel):
    """Complete hot stocks run with metadata and selections"""
    run_id: str
    run_timestamp: datetime
    metadata: HotStocksRunMetadata
    hot_stocks: List[HotStockSelection]
    summary: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class HotStocksRunCreateRequest(BaseModel):
    """Request to create a new hot stocks run"""
    universe_size: int = 50
    limit: int = 10
    min_momentum_pct: float = 0.1
    min_volume_spike: float = 0.01
    max_pe_ratio: float = 100.0
    min_roe: float = 5.0
    min_market_cap_cr: float = 500.0
    require_institutional: bool = False

class HotStocksRunResponse(BaseModel):
    """Response for hot stocks run"""
    run_id: str
    status: str
    message: str
    hot_stocks: List[HotStockSelection]
    metadata: HotStocksRunMetadata
    processing_time_seconds: float
    created_at: datetime
