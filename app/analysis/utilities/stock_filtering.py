import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def get_filtered_stocks(metrics: List[Dict[str, Any]], min_momentum_pct: float, min_volume_spike: float, require_institutional: bool, max_pe_ratio: float, min_roe: float, min_market_cap_cr: float, max_debt_equity: float = 0.5) -> bool:
    """Get filtered stocks based on criteria"""
    symbol = metrics.get("symbol", "UNKNOWN")
    momentum = metrics.get("momentum_pct_5d", 0.0)
    volume_spike = metrics.get("volume_spike_ratio", 0.0)
    institutional = metrics.get("institutional", False)
    
    # Use enhanced fundamentals if available, fallback to basic fundamentals
    enhanced_fundamentals = metrics.get("enhanced_fundamentals", {})
    basic_fundamentals = metrics.get("fundamentals", {})
    
    # Extract fundamental metrics with proper fallback
    pe_ratio = enhanced_fundamentals.get("pe_ratio") or basic_fundamentals.get("pe_ratio")
    # ROE is stored in basic_fundamentals.roe (not enhanced_fundamentals)
    roe = basic_fundamentals.get("roe") or enhanced_fundamentals.get("roe")
    market_cap = enhanced_fundamentals.get("market_cap") or basic_fundamentals.get("market_cap", 0)
    
    # Convert ROE from decimal percentage to percentage (0.1 -> 10%)
    roe_percentage = roe * 100 if roe is not None else None
    market_cap_cr = market_cap / 10000000 if market_cap else 0  # Convert to crores
    
    # Extract debt-to-equity ratio from correct location
    debt_equity = enhanced_fundamentals.get("quality_metrics", {}).get("debt_equity_ratio") or basic_fundamentals.get("debt_equity_ratio") or basic_fundamentals.get("debt_to_equity_ratio")
    
    roe_display = f"{roe_percentage:.2f}" if roe_percentage is not None else "N/A"
    logger.info(f"[hot-stocks] Filtering {symbol}: mom={momentum:.2f}% vol={volume_spike:.2f}x inst={institutional} pe={pe_ratio} roe={roe_display}% d/e={debt_equity} mcap={market_cap_cr:.0f}cr")
    
    # Check momentum filter
    if momentum < min_momentum_pct:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - momentum {momentum:.2f}% < {min_momentum_pct}%")
        return False
    
    # Check volume filter
    if volume_spike < min_volume_spike:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - volume spike {volume_spike:.2f}x < {min_volume_spike}x")
        return False
    
    # Check institutional filter
    if require_institutional and not institutional:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - no institutional activity")
        return False
    
    # Check fundamental filters
    if pe_ratio and pe_ratio > max_pe_ratio:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - P/E {pe_ratio:.1f} > {max_pe_ratio}")
        return False
    
    if roe_percentage and roe_percentage < min_roe:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - ROE {roe_percentage:.2f}% < {min_roe}%")
        return False
    
    if market_cap_cr < min_market_cap_cr:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - market cap {market_cap_cr:.0f}cr < {min_market_cap_cr}cr")
        return False
    
    # Check debt-to-equity filter
    if debt_equity and debt_equity > max_debt_equity:
        logger.info(f"[hot-stocks] ❌ {symbol}: REJECTED - debt/equity {debt_equity:.2f} > {max_debt_equity}")
        return False
    
    return True


# Rank by combined score: prioritize enhanced Stage 1 & 2 scores, fallback to basic scores
def score_stock(m: Dict[str, Any], use_enhanced_indicators: bool = True) -> float:
            # Calculate quality score boost based on fundamental strength
            enhanced_fundamentals = m.get("enhanced_fundamentals", {})
            basic_fundamentals = m.get("fundamentals", {})
            
            pe_ratio = enhanced_fundamentals.get("pe_ratio") or basic_fundamentals.get("pe_ratio") or 100
            # ROE is stored in basic_fundamentals.roe (not enhanced_fundamentals)
            roe = basic_fundamentals.get("roe") or enhanced_fundamentals.get("roe") or 0
            # D/E is stored in enhanced_fundamentals.quality_metrics.debt_equity_ratio
            debt_equity = enhanced_fundamentals.get("quality_metrics", {}).get("debt_equity_ratio") or basic_fundamentals.get("debt_equity_ratio") or basic_fundamentals.get("debt_to_equity_ratio") or 2.0
            
            # Quality scoring: 1.0 for good metrics, 0.5 for poor metrics
            pe_score = 1.0 if pe_ratio <= 25 else 0.5
            roe_score = 1.0 if roe >= 0.15 else 0.5
            debt_score = 1.0 if debt_equity <= 0.5 else 0.5
            
            quality_boost = (pe_score + roe_score + debt_score) / 3.0  # Average quality score (0.5-1.0)
            quality_boost_multi = 1.0 + quality_boost * 0.2  # Max 20% boost for quality (1.0-1.2)
            
            # PRIORITY 1: Use enhanced Stage 1 & 2 combined score if available
            enhanced_combined_score = m.get("enhanced_combined_score")
            if enhanced_combined_score is not None:
                # Use enhanced combined score with momentum and volume boosters
                momentum_boost = min(0.2, abs(m.get("momentum_pct_5d", 0.0)) / 50.0)  # Max 0.2 boost
                volume_boost = min(0.1, (m.get("volume_spike_ratio", 1.0) - 1.0) / 5.0)  # Max 0.1 boost
                enhanced_score = (enhanced_combined_score + momentum_boost + volume_boost) * quality_boost_multi
                return enhanced_score
            
            # PRIORITY 2: Use basic composite score if enhanced not available
            elif use_enhanced_indicators and "composite_score" in m:
                # Use composite technical score as primary, with momentum and volume as boosters
                composite = m.get("composite_score", 0.0)
                fundamental = m.get("fundamental_score", 0.5)  # Default neutral fundamental score
                momentum_boost = min(0.2, abs(m.get("momentum_pct_5d", 0.0)) / 50.0)  # Max 0.2 boost
                volume_boost = min(0.1, (m.get("volume_spike_ratio", 1.0) - 1.0) / 5.0)  # Max 0.1 boost
                
                # Combined score: 60% technical + 40% fundamental + boosters, with quality multiplier
                combined_score = ((0.6 * composite + 0.4 * fundamental) + momentum_boost + volume_boost) * quality_boost_multi
                return combined_score
            else:
                # Fallback to original scoring
                return m.get("momentum_pct_5d", 0.0) + 20.0 * (m.get("volume_spike_ratio", 0.0) - 1.0)
