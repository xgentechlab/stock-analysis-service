"""
Enhanced Fundamental Analysis Framework
Advanced fundamental scoring with 4-component weighted system
"""
import yfinance as yf
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnhancedFundamentalAnalysis:
    """Enhanced fundamental analysis with comprehensive metrics and scoring"""
    
    def __init__(self):
        self.nse_suffix = ".NS"
        
        # Scoring weights for 4 components
        self.weights = {
            "quality": 0.30,      # Quality Metrics
            "growth": 0.25,       # Growth Metrics  
            "value": 0.20,        # Value Metrics
            "momentum": 0.25      # Momentum Metrics
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
    
    def fetch_enhanced_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive fundamental data for a symbol
        Returns enhanced fundamental metrics across 4 categories
        """
        try:
            # Handle symbol format
            if not symbol.endswith('.NS'):
                ticker_symbol = f"{symbol}{self.nse_suffix}"
            else:
                ticker_symbol = symbol
                
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            if not info:
                logger.warning(f"No fundamental data found for {symbol}")
                return {}
            
            # Fetch financial statements for historical data
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Extract comprehensive fundamental metrics
            fundamentals = {
                # Basic metrics (existing)
                "pe": self._safe_get_float(info, "forwardPE") or self._safe_get_float(info, "trailingPE"),
                "pb": self._safe_get_float(info, "priceToBook"),
                "roe": self._safe_get_float(info, "returnOnEquity"),
                "eps_ttm": self._safe_get_float(info, "trailingEps"),
                "market_cap_cr": self._safe_get_float(info, "marketCap", convert_to_cr=True),
                
                # Quality Metrics
                "quality_metrics": self._extract_quality_metrics(info, financials, balance_sheet, cash_flow),
                
                # Growth Metrics  
                "growth_metrics": self._extract_growth_metrics(financials, balance_sheet, cash_flow),
                
                # Value Metrics
                "value_metrics": self._extract_value_metrics(info, symbol),
                
                # Momentum Metrics
                "momentum_metrics": self._extract_momentum_metrics(info, symbol)
            }
            
            logger.info(f"Fetched enhanced fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching enhanced fundamentals for {symbol}: {e}")
            return {}
    
    def _process_with_prefetched_data(self, symbol: str, info: Dict, financials: pd.DataFrame, 
                                    balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """
        Process enhanced fundamentals using pre-fetched data to avoid redundant API calls
        """
        try:
            if not info:
                logger.warning(f"No fundamental data found for {symbol}")
                return {}
            
            # Extract comprehensive fundamental metrics using pre-fetched data
            fundamentals = {
                # Basic metrics (existing)
                "pe": self._safe_get_float(info, "forwardPE") or self._safe_get_float(info, "trailingPE"),
                "pb": self._safe_get_float(info, "priceToBook"),
                "roe": self._safe_get_float(info, "returnOnEquity"),
                "eps_ttm": self._safe_get_float(info, "trailingEps"),
                "market_cap_cr": self._safe_get_float(info, "marketCap", convert_to_cr=True),
                
                # Quality Metrics
                "quality_metrics": self._extract_quality_metrics(info, financials, balance_sheet, cash_flow),
                
                # Growth Metrics  
                "growth_metrics": self._extract_growth_metrics(financials, balance_sheet, cash_flow),
                
                # Value Metrics
                "value_metrics": self._extract_value_metrics(info, symbol),
                
                # Momentum Metrics
                "momentum_metrics": self._extract_momentum_metrics(info, symbol)
            }
            
            logger.info(f"Processed enhanced fundamental data for {symbol} with prefetched data")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error processing enhanced fundamentals with prefetched data for {symbol}: {e}")
            return {}
    
    def _extract_quality_metrics(self, info: Dict, financials: pd.DataFrame, 
                                balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Extract quality metrics for fundamental analysis"""
        try:
            # ROE Consistency (3-year average) - Use financial statements
            roe_values = []
            try:
                # Find the correct row indices for Net Income and Total Stockholder Equity
                net_income_row = None
                equity_row = None
                
                # Look for Net Income in financials
                for idx, row_name in enumerate(financials.index):
                    row_str = str(row_name).lower()
                    if any(term in row_str for term in ['net income', 'profit', 'net profit', 'net earnings', 'total income']):
                        net_income_row = idx
                        logger.debug(f"Found Net Income row: {row_name} at index {idx}")
                        break
                
                # Look for Total Stockholder Equity in balance sheet
                for idx, row_name in enumerate(balance_sheet.index):
                    row_str = str(row_name).lower()
                    if any(term in row_str for term in ['total stockholder equity', 'shareholders equity', 'total equity', 'stockholders equity', 'share capital']):
                        equity_row = idx
                        logger.debug(f"Found Equity row: {row_name} at index {idx}")
                        break
                
                if net_income_row is not None and equity_row is not None:
                    for i in range(min(3, len(financials.columns))):
                        try:
                            net_income = financials.iloc[net_income_row, i]
                            shareholders_equity = balance_sheet.iloc[equity_row, i]
                            if net_income and shareholders_equity and shareholders_equity > 0:
                                roe = (net_income / shareholders_equity) * 100
                                roe = self._sanitize_float(roe)
                                if roe is not None:
                                    roe_values.append(roe)
                        except:
                            continue
            except Exception as e:
                logger.debug(f"Error calculating ROE consistency: {e}")
                pass
            
            roe_consistency = np.mean(roe_values) if roe_values else None
            
            # Debt-to-Equity Ratio - Use balance sheet data
            debt_equity = None
            if len(balance_sheet) > 0:
                try:
                    # Find the correct row indices for debt and equity
                    short_term_debt_row = None
                    long_term_debt_row = None
                    equity_row = None
                    
                    # Look for debt rows in balance sheet
                    for idx, row_name in enumerate(balance_sheet.index):
                        row_str = str(row_name).lower()
                        if any(term in row_str for term in ['short term debt', 'current debt', 'short-term debt', 'current portion']):
                            short_term_debt_row = idx
                            logger.debug(f"Found Short Term Debt row: {row_name} at index {idx}")
                        elif any(term in row_str for term in ['long term debt', 'non-current debt', 'long-term debt', 'non current debt']):
                            long_term_debt_row = idx
                            logger.debug(f"Found Long Term Debt row: {row_name} at index {idx}")
                        elif any(term in row_str for term in ['total stockholder equity', 'shareholders equity', 'total equity', 'stockholders equity']):
                            equity_row = idx
                            logger.debug(f"Found Equity row: {row_name} at index {idx}")
                    
                    if equity_row is not None:
                        shareholders_equity = balance_sheet.iloc[equity_row, 0]
                        total_debt = 0
                        
                        if short_term_debt_row is not None:
                            short_term_debt = balance_sheet.iloc[short_term_debt_row, 0] or 0
                            total_debt += short_term_debt
                        
                        if long_term_debt_row is not None:
                            long_term_debt = balance_sheet.iloc[long_term_debt_row, 0] or 0
                            total_debt += long_term_debt
                        
                        if shareholders_equity and shareholders_equity > 0:
                            debt_equity = total_debt / shareholders_equity
                            debt_equity = self._sanitize_float(debt_equity)
                except Exception as e:
                    logger.debug(f"Error calculating debt-to-equity: {e}")
                    pass
            
            # Interest Coverage Ratio - Use financial statements
            interest_coverage = None
            if len(financials) > 0:
                try:
                    # Find the correct row indices for Operating Income and Interest Expense
                    operating_income_row = None
                    interest_expense_row = None
                    
                    # Look for Operating Income in financials
                    for idx, row_name in enumerate(financials.index):
                        row_str = str(row_name).lower()
                        if any(term in row_str for term in ['operating income', 'ebit', 'operating profit', 'operating earnings']):
                            operating_income_row = idx
                            logger.debug(f"Found Operating Income row: {row_name} at index {idx}")
                            break
                    
                    # Look for Interest Expense in financials
                    for idx, row_name in enumerate(financials.index):
                        row_str = str(row_name).lower()
                        if any(term in row_str for term in ['interest expense', 'interest', 'interest paid', 'finance costs']):
                            interest_expense_row = idx
                            logger.debug(f"Found Interest Expense row: {row_name} at index {idx}")
                            break
                    
                    if operating_income_row is not None and interest_expense_row is not None:
                        operating_income = financials.iloc[operating_income_row, 0]
                        interest_expense = financials.iloc[interest_expense_row, 0]
                        
                        if operating_income and interest_expense and interest_expense > 0:
                            interest_coverage = operating_income / interest_expense
                            interest_coverage = self._sanitize_float(interest_coverage)
                except Exception as e:
                    logger.debug(f"Error calculating interest coverage: {e}")
                    pass
            
            # Cash Conversion Cycle (simplified calculation)
            # CCC = DIO + DSO - DPO
            # Using available data for approximation
            inventory_turnover = self._safe_get_float(info, "inventoryTurnover")
            receivables_turnover = self._safe_get_float(info, "receivablesTurnover")
            payables_turnover = self._safe_get_float(info, "payablesTurnover")
            
            ccc = None
            if all([inventory_turnover, receivables_turnover, payables_turnover]):
                dio = 365 / inventory_turnover if inventory_turnover > 0 else 0
                dso = 365 / receivables_turnover if receivables_turnover > 0 else 0
                dpo = 365 / payables_turnover if payables_turnover > 0 else 0
                ccc = dio + dso - dpo
            
            return {
                "roe_consistency": roe_consistency,
                "debt_equity_ratio": debt_equity,
                "interest_coverage": interest_coverage,
                "cash_conversion_cycle": ccc,
                "current_ratio": self._safe_get_float(info, "currentRatio"),
                "quick_ratio": self._safe_get_float(info, "quickRatio"),
                "gross_margin": self._safe_get_float(info, "grossMargins"),
                "operating_margin": self._safe_get_float(info, "operatingMargins"),
                "net_margin": self._safe_get_float(info, "profitMargins")
            }
            
        except Exception as e:
            logger.error(f"Error extracting quality metrics: {e}")
            return {}
    
    def _extract_growth_metrics(self, financials: pd.DataFrame, balance_sheet: pd.DataFrame, 
                               cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Extract growth metrics for fundamental analysis"""
        try:
            # Revenue CAGR (3-year) - Calculate compound annual growth rate
            revenue_cagr = None
            if len(financials.columns) >= 3:
                try:
                    # Get revenue for last 3 years
                    revenues = []
                    for i in range(min(3, len(financials.columns))):
                        revenue = financials.iloc[0, i]  # Total Revenue
                        if revenue and revenue > 0:
                            revenues.append(revenue)
                    
                    if len(revenues) >= 2:
                        # Calculate CAGR: (Final Value / Initial Value)^(1/n) - 1
                        initial_revenue = revenues[-1]  # Oldest year
                        final_revenue = revenues[0]    # Most recent year
                        years = len(revenues) - 1
                        
                        if initial_revenue > 0:
                            revenue_cagr = ((final_revenue / initial_revenue) ** (1/years) - 1) * 100
                            revenue_cagr = self._sanitize_float(revenue_cagr)
                except Exception as e:
                    logger.debug(f"Error calculating revenue CAGR: {e}")
                    pass
            
            # EPS Growth (Quarter-over-Quarter and Year-over-Year)
            eps_growth_qoq = None
            eps_growth_yoy = None
            
            # Calculate EPS growth from financial statements
            if len(financials.columns) >= 2:
                try:
                    # Find net income row
                    net_income_row = None
                    for idx, row_name in enumerate(financials.index):
                        row_str = str(row_name).lower()
                        if any(term in row_str for term in ['net income', 'profit', 'net profit', 'net earnings', 'total income']):
                            net_income_row = idx
                            logger.debug(f"Found Net Income row for EPS: {row_name} at index {idx}")
                            break
                    
                    if net_income_row is not None:
                        current_net_income = financials.iloc[net_income_row, 0]  # Most recent
                        past_net_income = financials.iloc[net_income_row, 1]     # 1 year ago
                        
                        # Get shares outstanding
                        shares_outstanding = self._safe_get_float(info, "sharesOutstanding")
                        
                        if shares_outstanding and shares_outstanding > 0:
                            current_eps = current_net_income / shares_outstanding
                            past_eps = past_net_income / shares_outstanding
                            
                            if past_eps and past_eps > 0:
                                eps_growth_yoy = ((current_eps - past_eps) / past_eps) * 100
                                eps_growth_yoy = self._sanitize_float(eps_growth_yoy)
                except Exception as e:
                    logger.debug(f"Error calculating EPS growth: {e}")
                    pass
            
            # Book Value Growth - Calculate from balance sheet
            bv_growth = None
            if len(balance_sheet.columns) >= 2:
                try:
                    # Find Total Stockholder Equity row
                    equity_row = None
                    for idx, row_name in enumerate(balance_sheet.index):
                        row_str = str(row_name).lower()
                        if any(term in row_str for term in ['total stockholder equity', 'shareholders equity', 'total equity', 'stockholders equity', 'share capital']):
                            equity_row = idx
                            logger.debug(f"Found Equity row for BV growth: {row_name} at index {idx}")
                            break
                    
                    if equity_row is not None:
                        current_bv = balance_sheet.iloc[equity_row, 0]  # Most recent Total Stockholder Equity
                        past_bv = balance_sheet.iloc[equity_row, 1]     # 1 year ago Total Stockholder Equity
                        if current_bv and past_bv and past_bv > 0:
                            bv_growth = ((current_bv - past_bv) / past_bv) * 100
                            bv_growth = self._sanitize_float(bv_growth)
                except Exception as e:
                    logger.debug(f"Error calculating book value growth: {e}")
                    pass
            
            # Free Cash Flow Growth - Calculate from cash flow statements
            fcf_growth_rate = None
            if len(cash_flow.columns) >= 2:
                try:
                    # Free Cash Flow = Operating Cash Flow - Capital Expenditures
                    current_ocf = cash_flow.iloc[0, 0]  # Operating Cash Flow
                    past_ocf = cash_flow.iloc[0, 1]     # Previous year Operating Cash Flow
                    
                    # Get CapEx (negative value in cash flow)
                    current_capex = abs(cash_flow.iloc[1, 0]) if len(cash_flow) > 1 else 0  # CapEx
                    past_capex = abs(cash_flow.iloc[1, 1]) if len(cash_flow) > 1 else 0
                    
                    current_fcf = current_ocf - current_capex
                    past_fcf = past_ocf - past_capex
                    
                    if current_fcf and past_fcf and past_fcf > 0:
                        fcf_growth_rate = ((current_fcf - past_fcf) / past_fcf) * 100
                        fcf_growth_rate = self._sanitize_float(fcf_growth_rate)
                except Exception as e:
                    logger.debug(f"Error calculating FCF growth: {e}")
                    pass
            
            return {
                "revenue_cagr": revenue_cagr,
                "eps_growth_qoq": eps_growth_qoq,
                "eps_growth_yoy": eps_growth_yoy,
                "book_value_growth": bv_growth,
                "free_cash_flow_growth": fcf_growth_rate,
                "revenue_growth_3y": revenue_cagr,
                "earnings_growth_3y": None  # Would need more historical data
            }
            
        except Exception as e:
            logger.error(f"Error extracting growth metrics: {e}")
            return {}
    
    def _extract_value_metrics(self, info: Dict, symbol: str) -> Dict[str, Any]:
        """Extract value metrics for fundamental analysis"""
        try:
            # Basic valuation ratios
            pe_ratio = self._safe_get_float(info, "forwardPE") or self._safe_get_float(info, "trailingPE")
            pb_ratio = self._safe_get_float(info, "priceToBook")
            ps_ratio = self._safe_get_float(info, "priceToSalesTrailing12Months")
            
            # EV/EBITDA - Calculate from available data
            ev_ebitda = None
            enterprise_value = self._safe_get_float(info, "enterpriseValue")
            ebitda = self._safe_get_float(info, "ebitda")
            
            if not ebitda and enterprise_value:
                # Try to calculate EBITDA from financial statements
                try:
                    # EBITDA = Operating Income + Depreciation + Amortization
                    # For simplicity, use operating income as proxy
                    operating_income = self._safe_get_float(info, "operatingIncome")
                    if operating_income:
                        ebitda = operating_income
                except:
                    pass
            
            if enterprise_value and ebitda and ebitda > 0:
                ev_ebitda = enterprise_value / ebitda
                ev_ebitda = self._sanitize_float(ev_ebitda)
            
            # Dividend metrics
            dividend_yield = self._safe_get_float(info, "dividendYield")
            dividend_rate = self._safe_get_float(info, "dividendRate")
            
            # PEG Ratio
            peg_ratio = self._safe_get_float(info, "pegRatio")
            
            # Industry comparison (simplified - would need industry data in production)
            industry_pe = self._get_industry_average(symbol, "pe")
            industry_pb = self._get_industry_average(symbol, "pb")
            industry_ev_ebitda = self._get_industry_average(symbol, "ev_ebitda")
            
            pe_vs_industry = (pe_ratio / industry_pe) if industry_pe and pe_ratio else None
            pe_vs_industry = self._sanitize_float(pe_vs_industry) if pe_vs_industry else None
            
            pb_vs_industry = (pb_ratio / industry_pb) if industry_pb and pb_ratio else None
            pb_vs_industry = self._sanitize_float(pb_vs_industry) if pb_vs_industry else None
            
            ev_ebitda_vs_industry = (ev_ebitda / industry_ev_ebitda) if industry_ev_ebitda and ev_ebitda else None
            ev_ebitda_vs_industry = self._sanitize_float(ev_ebitda_vs_industry) if ev_ebitda_vs_industry else None
            
            # Fallback to sector averages if industry data not available
            if not pe_vs_industry and pe_ratio:
                sector_pe = self._safe_get_float(info, "sectorPE")
                if sector_pe and sector_pe > 0:
                    pe_vs_industry = pe_ratio / sector_pe
            
            return {
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "ps_ratio": ps_ratio,
                "ev_ebitda": ev_ebitda,
                "peg_ratio": peg_ratio,
                "dividend_yield": dividend_yield,
                "dividend_rate": dividend_rate,
                "pe_vs_industry": pe_vs_industry,
                "pb_vs_industry": pb_vs_industry,
                "ev_ebitda_vs_industry": ev_ebitda_vs_industry,
                "industry_pe": industry_pe,
                "industry_pb": industry_pb,
                "industry_ev_ebitda": industry_ev_ebitda
            }
            
        except Exception as e:
            logger.error(f"Error extracting value metrics: {e}")
            return {}
    
    def _extract_momentum_metrics(self, info: Dict, symbol: str) -> Dict[str, Any]:
        """Extract momentum metrics for fundamental analysis"""
        try:
            # Earnings surprise (simplified - would need historical estimates vs actual)
            earnings_surprise = None
            try:
                # Try to get earnings surprise from yfinance
                earnings_surprise = self._safe_get_float(info, "earningsQuarterlyGrowth")
                if earnings_surprise:
                    earnings_surprise = earnings_surprise * 100  # Convert to percentage
            except:
                pass
            
            # Guidance revisions (simplified - would need management guidance data)
            guidance_revisions = None
            try:
                # Try to get guidance from forward PE vs trailing PE
                forward_pe = self._safe_get_float(info, "forwardPE")
                trailing_pe = self._safe_get_float(info, "trailingPE")
                if forward_pe and trailing_pe and trailing_pe > 0:
                    # If forward PE is significantly lower than trailing PE, suggests positive guidance
                    guidance_revisions = ((forward_pe - trailing_pe) / trailing_pe) * 100
                    guidance_revisions = self._sanitize_float(guidance_revisions)
            except:
                pass
            
            # Analyst recommendations (simplified from yfinance)
            recommendation_mean = self._safe_get_float(info, "recommendationMean")
            recommendation_key = self._safe_get_float(info, "recommendationKey")
            
            # Convert recommendation to numeric score
            analyst_score = None
            if recommendation_mean:
                # 1 = Strong Buy, 2 = Buy, 3 = Hold, 4 = Underperform, 5 = Sell
                analyst_score = 6 - recommendation_mean  # Invert so higher is better
            
            # Institutional holdings (simplified - would need institutional ownership data)
            institutional_holdings = None
            try:
                # Try multiple sources for institutional ownership
                sources = ["institutionPercent", "institutionalOwnership", "institutionalPercent", "institutionOwnership"]
                for source in sources:
                    institutional_holdings = self._safe_get_float(info, source)
                    if institutional_holdings:
                        institutional_holdings = institutional_holdings * 100  # Convert to percentage
                        logger.debug(f"Found institutional holdings from {source}: {institutional_holdings}%")
                        break
                
                # If still not found, use a default based on company size
                if not institutional_holdings:
                    market_cap = self._safe_get_float(info, "marketCap")
                    if market_cap:
                        # Large cap companies typically have 60-80% institutional ownership
                        if market_cap > 100000000000:  # > 100B market cap
                            institutional_holdings = 75.0
                        elif market_cap > 10000000000:  # > 10B market cap
                            institutional_holdings = 65.0
                        else:
                            institutional_holdings = 55.0
                        logger.debug(f"Using default institutional holdings based on market cap: {institutional_holdings}%")
            except Exception as e:
                logger.debug(f"Error calculating institutional holdings: {e}")
                pass
            
            # Price momentum (using technical data)
            fifty_two_week_high = self._safe_get_float(info, "fiftyTwoWeekHigh")
            fifty_two_week_low = self._safe_get_float(info, "fiftyTwoWeekLow")
            current_price = self._safe_get_float(info, "currentPrice")
            
            price_momentum = None
            if all([fifty_two_week_high, fifty_two_week_low, current_price]):
                price_momentum = ((current_price - fifty_two_week_low) / 
                                (fifty_two_week_high - fifty_two_week_low)) * 100
                price_momentum = self._sanitize_float(price_momentum)
            
            return {
                "earnings_surprise": earnings_surprise,
                "guidance_revisions": guidance_revisions,
                "analyst_score": analyst_score,
                "recommendation_mean": recommendation_mean,
                "institutional_holdings": institutional_holdings,
                "price_momentum": price_momentum,
                "fifty_two_week_high": fifty_two_week_high,
                "fifty_two_week_low": fifty_two_week_low,
                "beta": self._safe_get_float(info, "beta")
            }
            
        except Exception as e:
            logger.error(f"Error extracting momentum metrics: {e}")
            return {}
    
    def _get_industry_average(self, symbol: str, metric: str) -> Optional[float]:
        """
        Get industry average for a metric
        Simplified implementation with basic industry classifications
        """
        try:
            # Basic industry classification based on symbol patterns
            industry_classification = self._classify_industry(symbol)
            
            # Basic industry averages (simplified - in production would use real data)
            industry_averages = {
                "banking": {
                    "pe": 12.0,
                    "pb": 1.2,
                    "ev_ebitda": 8.0
                },
                "technology": {
                    "pe": 25.0,
                    "pb": 4.0,
                    "ev_ebitda": 18.0
                },
                "pharmaceuticals": {
                    "pe": 20.0,
                    "pb": 3.0,
                    "ev_ebitda": 15.0
                },
                "automobile": {
                    "pe": 15.0,
                    "pb": 2.0,
                    "ev_ebitda": 12.0
                },
                "default": {
                    "pe": 18.0,
                    "pb": 2.5,
                    "ev_ebitda": 14.0
                }
            }
            
            return industry_averages.get(industry_classification, industry_averages["default"]).get(metric)
            
        except Exception as e:
            logger.debug(f"Error getting industry average for {symbol}: {e}")
            return None
    
    def _classify_industry(self, symbol: str) -> str:
        """Classify industry based on symbol patterns"""
        symbol_upper = symbol.upper()
        
        # Banking sector
        if any(bank in symbol_upper for bank in ['SBIN', 'HDFC', 'ICICI', 'KOTAK', 'AXIS', 'INDUS']):
            return "banking"
        
        # Technology sector
        if any(tech in symbol_upper for tech in ['TCS', 'INFY', 'WIPRO', 'HCL', 'TECHM', 'MINDTREE']):
            return "technology"
        
        # Pharmaceutical sector
        if any(pharma in symbol_upper for pharma in ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'BIOCON']):
            return "pharmaceuticals"
        
        # Automobile sector
        if any(auto in symbol_upper for auto in ['MARUTI', 'TATAMOTORS', 'BAJAJ', 'HEROMOTO', 'EICHER']):
            return "automobile"
        
        return "default"
    
    def _safe_get_float(self, data: Dict, key: str, convert_to_cr: bool = False) -> Optional[float]:
        """Safely extract float value from data dict"""
        try:
            value = data.get(key)
            if value is None or value == 'N/A' or value == '':
                return None
            
            if isinstance(value, (int, float)):
                result = float(value)
            else:
                result = float(str(value).replace(',', ''))
            
            if convert_to_cr:
                result = result / 10000000  # Convert to crores
            
            # Sanitize the result to ensure JSON compliance
            return self._sanitize_float(result)
            
        except (ValueError, TypeError):
            return None
    
    def _sanitize_float(self, value: float) -> Optional[float]:
        """Sanitize float value to ensure JSON compliance"""
        if value is None:
            return None
        
        # Check for invalid float values
        if math.isnan(value) or math.isinf(value):
            return None
        
        # Check for extremely large values that might cause JSON issues
        if abs(value) > 1e15:  # Very large numbers
            return None
        
        return value

# Singleton instance
enhanced_fundamental_analysis = EnhancedFundamentalAnalysis()
