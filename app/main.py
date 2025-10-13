"""
FastAPI main application
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
import time
from contextlib import asynccontextmanager

from app.config import settings
from app.api import routes_signals, routes_positions, routes_admin, routes_jobs
from app.models.schemas import ApiResponse

# Configure logging - Use INFO level to reduce log noise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce noise from specific libraries
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Stock Selection Backend API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Test Firestore connection
    try:
        from app.db.firestore_client import firestore_client
        config = firestore_client.get_runtime_config()
        logger.info("Firestore connection successful")
        logger.info(f"Paper mode: {config.get('paper_mode', True)}")
        logger.info(f"Kill switch: {config.get('kill_switch', False)}")
    except Exception as e:
        logger.error(f"Firestore connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stock Selection Backend API")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Stock selection and trading backend API",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            ok=False,
            error="Internal server error"
        ).model_dump()
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Firestore connection
        from app.db.firestore_client import firestore_client
        firestore_client.get_runtime_config()
        
        return ApiResponse(
            ok=True,
            data={
                "status": "healthy",
                "environment": settings.environment,
                "version": settings.app_version
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ApiResponse(
            ok=False,
            error="Service unhealthy"
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return ApiResponse(
        ok=True,
        data={
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "environment": settings.environment,
            "docs_url": "/docs"
        }
    )

# Include routers
app.include_router(routes_signals.router)
app.include_router(routes_positions.router)
app.include_router(routes_admin.router)
app.include_router(routes_jobs.router)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    import time
    import os
    
    # Use port from settings
    port = settings.port
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )
