import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = True  # Default to False, will be validated below
    
    @field_validator('debug', mode='before')
    @classmethod
    def validate_debug(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)
    
    # FastAPI
    app_name: str = "Stock Selection Backend"
    app_version: str = "1.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    
    # Firestore
    firestore_project_id: str = os.getenv("FIRESTORE_PROJECT_ID", "")
    firestore_database_id: str = os.getenv("FIRESTORE_DATABASE_ID", "stock-data")
    google_application_credentials: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Anthropic Claude
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    claude_max_tokens: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4000"))
    claude_temperature: float = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))
    
    # AI Provider Selection
    ai_provider: str = os.getenv("AI_PROVIDER", "claude")  # "openai" or "claude"
    
    # Admin endpoints (no authentication required for development)
    # admin_token removed as per user request
    
    # Stock selection parameters
    universe_size: int = int(os.getenv("UNIVERSE_SIZE", "200"))
    daily_candidates: int = int(os.getenv("DAILY_CANDIDATES", "40"))
    openai_shortlist: int = int(os.getenv("OPENAI_SHORTLIST", "15"))
    min_signal_score: float = float(os.getenv("MIN_SIGNAL_SCORE", "0.5"))
    top_pick_score: float = float(os.getenv("TOP_PICK_SCORE", "0.7"))
    
    # Scoring weights
    momentum_weight: float = float(os.getenv("MOMENTUM_WEIGHT", "0.45"))
    volume_weight: float = float(os.getenv("VOLUME_WEIGHT", "0.35"))
    breakout_weight: float = float(os.getenv("BREAKOUT_WEIGHT", "0.2"))
    
    # Volume spike parameters
    volume_threshold: float = float(os.getenv("VOLUME_THRESHOLD", "2.0"))
    volume_cap: float = float(os.getenv("VOLUME_CAP", "5.0"))
    
    # Fundamental filters
    min_market_cap_cr: float = float(os.getenv("MIN_MARKET_CAP_CR", "500"))
    max_pe_ratio: float = float(os.getenv("MAX_PE_RATIO", "60"))
    
    # Expanded universe settings
    expanded_universe_size: int = int(os.getenv("EXPANDED_UNIVERSE_SIZE", "500"))
    enable_enhanced_indicators: bool = bool(os.getenv("ENABLE_ENHANCED_INDICATORS", "true"))
    technical_score_weights: str = os.getenv("TECHNICAL_SCORE_WEIGHTS", "0.2,0.2,0.15,0.2,0.15,0.1")  # momentum,volume,breakout,trend,momentum_osc,volume_momentum
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env file

settings = Settings()
