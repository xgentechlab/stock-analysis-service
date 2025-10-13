"""
Fundamental Scoring Engine
4-component weighted scoring system for fundamental analysis
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FundamentalScoringEngine:
    """Fundamental scoring engine with 4-component weighted system"""
    
    def __init__(self):
        # Scoring weights for 4 components
        self.weights = {
            "quality": 0.30,      # Quality Metrics (30%)
            "growth": 0.25,       # Growth Metrics (25%)
            "value": 0.20,        # Value Metrics (20%)
            "momentum": 0.25      # Momentum Metrics (25%)
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "roe_consistency_min": 15.0,      # 3-year average ROE > 15%
            "debt_equity_max": 0.5,           # Debt-to-Equity < 0.5
            "interest_coverage_min": 3.0,     # Interest Coverage > 3x
            "ccc_max": 90.0                   # Cash Conversion Cycle < 90 days
        }
        
        # Growth thresholds
        self.growth_thresholds = {
            "revenue_cagr_min": 10.0,         # Revenue CAGR > 10%
            "eps_growth_min": 15.0,           # EPS Growth > 15%
            "book_value_growth_min": 8.0,     # Book Value Growth > 8%
            "fcf_growth_min": 12.0            # Free Cash Flow Growth > 12%
        }
        
        # Value thresholds (relative to industry)
        self.value_thresholds = {
            "pe_vs_industry_max": 1.2,        # P/E < 1.2x industry average
            "pb_vs_industry_max": 1.1,        # P/B < 1.1x industry average
            "ev_ebitda_vs_industry_max": 1.3, # EV/EBITDA < 1.3x industry average
            "dividend_yield_min": 1.0         # Dividend Yield > 1%
        }
        
        # Momentum thresholds
        self.momentum_thresholds = {
            "earnings_surprise_min": 0.05,    # Earnings beat by > 5%
            "guidance_revisions_min": 0.02,   # Positive guidance revisions > 2%
            "analyst_upgrades_min": 0.1,      # Analyst upgrade ratio > 10%
            "institutional_holding_min": 0.05 # Institutional holding increase > 5%
        }
    
    def calculate_fundamental_score(self, fundamental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate fundamental score using 4-component weighted system
        Returns comprehensive scoring breakdown
        """
        try:
            scores = {}
            
            # 1. Quality Score (30%)
            quality_score = self._calculate_quality_score(fundamental_data)
            scores["quality"] = quality_score
            
            # 2. Growth Score (25%)
            growth_score = self._calculate_growth_score(fundamental_data)
            scores["growth"] = growth_score
            
            # 3. Value Score (20%)
            value_score = self._calculate_value_score(fundamental_data)
            scores["value"] = value_score
            
            # 4. Momentum Score (25%)
            momentum_score = self._calculate_momentum_score(fundamental_data)
            scores["momentum"] = momentum_score
            
            # Calculate weighted final score
            final_score = sum(
                self.weights[component] * score 
                for component, score in scores.items()
            )
            
            # Determine fundamental strength
            fundamental_strength = self._determine_fundamental_strength(final_score)
            
            return {
                "final_score": final_score,
                "fundamental_strength": fundamental_strength,
                "component_scores": scores,
                "weights": self.weights,
                "confidence": self._calculate_confidence(scores, fundamental_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return {"final_score": 0.0, "fundamental_strength": "poor", "error": str(e)}
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate quality score based on financial health metrics"""
        try:
            quality_metrics = data.get("quality_metrics", {})
            score_components = []
            
            # ROE Consistency (0-1 score)
            roe_consistency = quality_metrics.get("roe_consistency")
            if roe_consistency is not None:
                # Score: 1.0 if > 20%, 0.8 if > 15%, 0.6 if > 10%, 0.4 if > 5%, 0.0 if < 5%
                if roe_consistency >= 20:
                    roe_score = 1.0
                elif roe_consistency >= 15:
                    roe_score = 0.8
                elif roe_consistency >= 10:
                    roe_score = 0.6
                elif roe_consistency >= 5:
                    roe_score = 0.4
                else:
                    roe_score = 0.0
                score_components.append(roe_score)
            
            # Debt-to-Equity Ratio (0-1 score, lower is better)
            debt_equity = quality_metrics.get("debt_equity_ratio")
            if debt_equity is not None:
                # Score: 1.0 if < 0.3, 0.8 if < 0.5, 0.6 if < 0.7, 0.4 if < 1.0, 0.0 if > 1.0
                if debt_equity <= 0.3:
                    debt_score = 1.0
                elif debt_equity <= 0.5:
                    debt_score = 0.8
                elif debt_equity <= 0.7:
                    debt_score = 0.6
                elif debt_equity <= 1.0:
                    debt_score = 0.4
                else:
                    debt_score = 0.0
                score_components.append(debt_score)
            
            # Interest Coverage Ratio (0-1 score, higher is better)
            interest_coverage = quality_metrics.get("interest_coverage")
            if interest_coverage is not None:
                # Score: 1.0 if > 5x, 0.8 if > 3x, 0.6 if > 2x, 0.4 if > 1x, 0.0 if < 1x
                if interest_coverage >= 5:
                    coverage_score = 1.0
                elif interest_coverage >= 3:
                    coverage_score = 0.8
                elif interest_coverage >= 2:
                    coverage_score = 0.6
                elif interest_coverage >= 1:
                    coverage_score = 0.4
                else:
                    coverage_score = 0.0
                score_components.append(coverage_score)
            
            # Cash Conversion Cycle (0-1 score, lower is better)
            ccc = quality_metrics.get("cash_conversion_cycle")
            if ccc is not None:
                # Score: 1.0 if < 30 days, 0.8 if < 60 days, 0.6 if < 90 days, 0.4 if < 120 days, 0.0 if > 120 days
                if ccc <= 30:
                    ccc_score = 1.0
                elif ccc <= 60:
                    ccc_score = 0.8
                elif ccc <= 90:
                    ccc_score = 0.6
                elif ccc <= 120:
                    ccc_score = 0.4
                else:
                    ccc_score = 0.0
                score_components.append(ccc_score)
            
            # Profitability Margins
            gross_margin = quality_metrics.get("gross_margin")
            operating_margin = quality_metrics.get("operating_margin")
            net_margin = quality_metrics.get("net_margin")
            
            margin_scores = []
            for margin in [gross_margin, operating_margin, net_margin]:
                if margin is not None:
                    # Convert to percentage and score
                    margin_pct = margin * 100
                    if margin_pct >= 20:
                        margin_scores.append(1.0)
                    elif margin_pct >= 15:
                        margin_scores.append(0.8)
                    elif margin_pct >= 10:
                        margin_scores.append(0.6)
                    elif margin_pct >= 5:
                        margin_scores.append(0.4)
                    else:
                        margin_scores.append(0.0)
            
            if margin_scores:
                score_components.append(np.mean(margin_scores))
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _calculate_growth_score(self, data: Dict[str, Any]) -> float:
        """Calculate growth score based on growth metrics"""
        try:
            growth_metrics = data.get("growth_metrics", {})
            score_components = []
            
            # Revenue CAGR (0-1 score)
            revenue_cagr = growth_metrics.get("revenue_cagr")
            if revenue_cagr is not None:
                # Score: 1.0 if > 20%, 0.8 if > 15%, 0.6 if > 10%, 0.4 if > 5%, 0.0 if < 0%
                if revenue_cagr >= 20:
                    revenue_score = 1.0
                elif revenue_cagr >= 15:
                    revenue_score = 0.8
                elif revenue_cagr >= 10:
                    revenue_score = 0.6
                elif revenue_cagr >= 5:
                    revenue_score = 0.4
                elif revenue_cagr >= 0:
                    revenue_score = 0.2
                else:
                    revenue_score = 0.0
                score_components.append(revenue_score)
            
            # Book Value Growth (0-1 score)
            bv_growth = growth_metrics.get("book_value_growth")
            if bv_growth is not None:
                # Score: 1.0 if > 15%, 0.8 if > 10%, 0.6 if > 5%, 0.4 if > 0%, 0.0 if < 0%
                if bv_growth >= 15:
                    bv_score = 1.0
                elif bv_growth >= 10:
                    bv_score = 0.8
                elif bv_growth >= 5:
                    bv_score = 0.6
                elif bv_growth >= 0:
                    bv_score = 0.4
                else:
                    bv_score = 0.0
                score_components.append(bv_score)
            
            # Free Cash Flow Growth (0-1 score)
            fcf_growth = growth_metrics.get("free_cash_flow_growth")
            if fcf_growth is not None:
                # Score: 1.0 if > 20%, 0.8 if > 15%, 0.6 if > 10%, 0.4 if > 5%, 0.0 if < 0%
                if fcf_growth >= 20:
                    fcf_score = 1.0
                elif fcf_growth >= 15:
                    fcf_score = 0.8
                elif fcf_growth >= 10:
                    fcf_score = 0.6
                elif fcf_growth >= 5:
                    fcf_score = 0.4
                elif fcf_growth >= 0:
                    fcf_score = 0.2
                else:
                    fcf_score = 0.0
                score_components.append(fcf_score)
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating growth score: {e}")
            return 0.5
    
    def _calculate_value_score(self, data: Dict[str, Any]) -> float:
        """Calculate value score based on valuation metrics"""
        try:
            value_metrics = data.get("value_metrics", {})
            score_components = []
            
            # P/E vs Industry (0-1 score, lower is better)
            pe_vs_industry = value_metrics.get("pe_vs_industry")
            if pe_vs_industry is not None:
                # Score: 1.0 if < 0.8x, 0.8 if < 1.0x, 0.6 if < 1.2x, 0.4 if < 1.5x, 0.0 if > 1.5x
                if pe_vs_industry <= 0.8:
                    pe_score = 1.0
                elif pe_vs_industry <= 1.0:
                    pe_score = 0.8
                elif pe_vs_industry <= 1.2:
                    pe_score = 0.6
                elif pe_vs_industry <= 1.5:
                    pe_score = 0.4
                else:
                    pe_score = 0.0
                score_components.append(pe_score)
            
            # P/B vs Industry (0-1 score, lower is better)
            pb_vs_industry = value_metrics.get("pb_vs_industry")
            if pb_vs_industry is not None:
                # Score: 1.0 if < 0.8x, 0.8 if < 1.0x, 0.6 if < 1.2x, 0.4 if < 1.5x, 0.0 if > 1.5x
                if pb_vs_industry <= 0.8:
                    pb_score = 1.0
                elif pb_vs_industry <= 1.0:
                    pb_score = 0.8
                elif pb_vs_industry <= 1.2:
                    pb_score = 0.6
                elif pb_vs_industry <= 1.5:
                    pb_score = 0.4
                else:
                    pb_score = 0.0
                score_components.append(pb_score)
            
            # EV/EBITDA vs Industry (0-1 score, lower is better)
            ev_ebitda_vs_industry = value_metrics.get("ev_ebitda_vs_industry")
            if ev_ebitda_vs_industry is not None:
                # Score: 1.0 if < 0.8x, 0.8 if < 1.0x, 0.6 if < 1.3x, 0.4 if < 1.8x, 0.0 if > 1.8x
                if ev_ebitda_vs_industry <= 0.8:
                    ev_score = 1.0
                elif ev_ebitda_vs_industry <= 1.0:
                    ev_score = 0.8
                elif ev_ebitda_vs_industry <= 1.3:
                    ev_score = 0.6
                elif ev_ebitda_vs_industry <= 1.8:
                    ev_score = 0.4
                else:
                    ev_score = 0.0
                score_components.append(ev_score)
            
            # Dividend Yield (0-1 score, higher is better)
            dividend_yield = value_metrics.get("dividend_yield")
            if dividend_yield is not None:
                # Score: 1.0 if > 4%, 0.8 if > 3%, 0.6 if > 2%, 0.4 if > 1%, 0.0 if < 1%
                if dividend_yield >= 4:
                    div_score = 1.0
                elif dividend_yield >= 3:
                    div_score = 0.8
                elif dividend_yield >= 2:
                    div_score = 0.6
                elif dividend_yield >= 1:
                    div_score = 0.4
                else:
                    div_score = 0.0
                score_components.append(div_score)
            
            # Absolute P/E Ratio (fallback if industry comparison not available)
            pe_ratio = data.get("pe")
            if pe_ratio is not None and not pe_vs_industry:
                # Score: 1.0 if < 15, 0.8 if < 20, 0.6 if < 25, 0.4 if < 30, 0.0 if > 30
                if pe_ratio <= 15:
                    pe_abs_score = 1.0
                elif pe_ratio <= 20:
                    pe_abs_score = 0.8
                elif pe_ratio <= 25:
                    pe_abs_score = 0.6
                elif pe_ratio <= 30:
                    pe_abs_score = 0.4
                else:
                    pe_abs_score = 0.0
                score_components.append(pe_abs_score)
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating value score: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, data: Dict[str, Any]) -> float:
        """Calculate momentum score based on momentum metrics"""
        try:
            momentum_metrics = data.get("momentum_metrics", {})
            score_components = []
            
            # Analyst Score (0-1 score, higher is better)
            analyst_score = momentum_metrics.get("analyst_score")
            if analyst_score is not None:
                # Convert 1-5 scale to 0-1 scale (inverted)
                score_components.append(analyst_score / 5.0)
            
            # Price Momentum (0-1 score)
            price_momentum = momentum_metrics.get("price_momentum")
            if price_momentum is not None:
                # Score based on position within 52-week range
                # 1.0 if > 80%, 0.8 if > 60%, 0.6 if > 40%, 0.4 if > 20%, 0.0 if < 20%
                if price_momentum >= 80:
                    momentum_score = 1.0
                elif price_momentum >= 60:
                    momentum_score = 0.8
                elif price_momentum >= 40:
                    momentum_score = 0.6
                elif price_momentum >= 20:
                    momentum_score = 0.4
                else:
                    momentum_score = 0.0
                score_components.append(momentum_score)
            
            # Beta (0-1 score, moderate beta is better)
            beta = momentum_metrics.get("beta")
            if beta is not None:
                # Score: 1.0 if beta 0.8-1.2, 0.8 if 0.6-1.4, 0.6 if 0.4-1.6, 0.4 if 0.2-1.8, 0.0 if extreme
                if 0.8 <= beta <= 1.2:
                    beta_score = 1.0
                elif 0.6 <= beta <= 1.4:
                    beta_score = 0.8
                elif 0.4 <= beta <= 1.6:
                    beta_score = 0.6
                elif 0.2 <= beta <= 1.8:
                    beta_score = 0.4
                else:
                    beta_score = 0.0
                score_components.append(beta_score)
            
            # Earnings Surprise (if available)
            earnings_surprise = momentum_metrics.get("earnings_surprise")
            if earnings_surprise is not None:
                # Score: 1.0 if > 10%, 0.8 if > 5%, 0.6 if > 0%, 0.4 if > -5%, 0.0 if < -5%
                if earnings_surprise >= 10:
                    surprise_score = 1.0
                elif earnings_surprise >= 5:
                    surprise_score = 0.8
                elif earnings_surprise >= 0:
                    surprise_score = 0.6
                elif earnings_surprise >= -5:
                    surprise_score = 0.4
                else:
                    surprise_score = 0.0
                score_components.append(surprise_score)
            
            return float(np.mean(score_components)) if score_components else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    def _determine_fundamental_strength(self, final_score: float) -> str:
        """Determine fundamental strength based on final score"""
        if final_score >= 0.8:
            return "excellent"
        elif final_score >= 0.6:
            return "good"
        elif final_score >= 0.4:
            return "average"
        elif final_score >= 0.2:
            return "poor"
        else:
            return "very_poor"
    
    def _calculate_confidence(self, scores: Dict[str, float], data: Dict[str, Any]) -> float:
        """Calculate confidence level based on score consistency and data quality"""
        try:
            # Score consistency (lower std = higher confidence)
            score_values = list(scores.values())
            consistency = 1.0 - float(np.std(score_values))
            
            # Data quality indicators
            quality_indicators = []
            
            # Check if we have all major fundamental categories
            major_categories = ["quality_metrics", "growth_metrics", "value_metrics", "momentum_metrics"]
            available_categories = sum(1 for category in major_categories if data.get(category))
            quality_indicators.append(available_categories / len(major_categories))
            
            # Check for key metrics availability
            key_metrics = ["pe", "pb", "roe", "market_cap_cr"]
            available_metrics = sum(1 for metric in key_metrics if data.get(metric) is not None)
            quality_indicators.append(available_metrics / len(key_metrics))
            
            data_quality = float(np.mean(quality_indicators)) if quality_indicators else 0.5
            
            # Combine consistency and data quality
            confidence = (consistency * 0.6 + data_quality * 0.4)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_fundamental_breakdown(self, fundamental_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed fundamental breakdown for analysis"""
        fundamental_score = self.calculate_fundamental_score(fundamental_data)
        
        return {
            "fundamental_score": fundamental_score,
            "scoring_weights": self.weights,
            "thresholds": {
                "quality": self.quality_thresholds,
                "growth": self.growth_thresholds,
                "value": self.value_thresholds,
                "momentum": self.momentum_thresholds
            },
            "recommendations": self._get_fundamental_recommendations(fundamental_score, fundamental_data)
        }
    
    def _get_fundamental_recommendations(self, score_data: Dict[str, Any], fundamental_data: Dict[str, Any]) -> List[str]:
        """Get fundamental analysis recommendations"""
        recommendations = []
        
        fundamental_strength = score_data.get("fundamental_strength", "average")
        confidence = score_data.get("confidence", 0.5)
        
        if fundamental_strength in ["excellent", "good"] and confidence > 0.7:
            recommendations.append("Strong fundamental profile - suitable for long-term investment")
        
        if fundamental_strength in ["very_poor", "poor"] and confidence > 0.7:
            recommendations.append("Weak fundamental profile - avoid or monitor closely")
        
        # Quality recommendations
        quality_metrics = fundamental_data.get("quality_metrics", {})
        if quality_metrics.get("debt_equity_ratio", 0) > 0.7:
            recommendations.append("High debt levels - monitor debt management")
        
        if quality_metrics.get("roe_consistency", 0) < 10:
            recommendations.append("Low ROE consistency - check profitability trends")
        
        # Growth recommendations
        growth_metrics = fundamental_data.get("growth_metrics", {})
        if growth_metrics.get("revenue_cagr", 0) < 5:
            recommendations.append("Slow revenue growth - verify business model sustainability")
        
        # Value recommendations
        value_metrics = fundamental_data.get("value_metrics", {})
        if value_metrics.get("pe_ratio", 0) > 30:
            recommendations.append("High P/E ratio - verify growth expectations")
        
        return recommendations

# Singleton instance
fundamental_scoring = FundamentalScoringEngine()
