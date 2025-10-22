#!/usr/bin/env python3
"""
Script to populate Nifty 500 stocks from CSV file into Firestore
"""
import csv
import sys
import os
import logging
from typing import List, Dict, Any
import argparse

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from db.firestore_client import firestore_client
from models.schemas import Stock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_nifty500_csv(csv_file_path: str) -> List[Dict[str, Any]]:
    """Read and parse the Nifty 500 CSV file"""
    stocks = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):  # Start from 2 since header is row 1
                try:
                    # Skip empty rows
                    if not row.get('Company Name') or not row.get('Symbol'):
                        continue
                    
                    # Clean and prepare stock data
                    stock_data = {
                        'company_name': row['Company Name'].strip(),
                        'symbol': row['Symbol'].strip(),
                        'industry': row['Industry'].strip(),
                        'series': row.get('Series', '').strip() or None,
                        'isin_code': row.get('ISIN Code', '').strip() or None
                    }
                    
                    # Validate required fields
                    if not stock_data['company_name'] or not stock_data['symbol'] or not stock_data['industry']:
                        logger.warning(f"Row {row_num}: Skipping incomplete data - {stock_data}")
                        continue
                    
                    stocks.append(stock_data)
                    
                except Exception as e:
                    logger.error(f"Row {row_num}: Error processing row - {e}")
                    continue
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    logger.info(f"Successfully read {len(stocks)} stocks from CSV file")
    return stocks

def clear_existing_stocks(confirm: bool = False) -> bool:
    """Clear existing stocks from database"""
    if not confirm:
        response = input("Are you sure you want to clear all existing stocks? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Operation cancelled")
            return False
    
    try:
        # Get all stocks
        result = firestore_client.list_stocks(is_active=None, limit=10000)
        stocks = result.get('stocks', [])
        
        if not stocks:
            logger.info("No existing stocks found")
            return True
        
        logger.info(f"Found {len(stocks)} existing stocks")
        
        # Soft delete all stocks
        deleted_count = 0
        for stock in stocks:
            try:
                if firestore_client.delete_stock(stock['id']):
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete stock {stock.get('symbol')}: {e}")
        
        logger.info(f"Soft deleted {deleted_count} existing stocks")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing existing stocks: {e}")
        return False

def populate_stocks(stocks_data: List[Dict[str, Any]], clear_existing: bool = False) -> Dict[str, Any]:
    """Populate stocks in the database"""
    try:
        # Clear existing stocks if requested
        if clear_existing:
            if not clear_existing_stocks(confirm=True):
                return {"error": "Failed to clear existing stocks"}
        
        # Check for existing stocks
        existing_result = firestore_client.list_stocks(is_active=True, limit=1)
        if existing_result.get('total', 0) > 0 and not clear_existing:
            logger.warning(f"Found {existing_result['total']} existing stocks. Use --clear to replace them.")
            return {"error": "Existing stocks found. Use --clear flag to replace them."}
        
        # Bulk create stocks
        logger.info(f"Creating {len(stocks_data)} stocks...")
        result = firestore_client.bulk_create_stocks(stocks_data)
        
        logger.info(f"Bulk create result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error populating stocks: {e}")
        return {"error": str(e)}

def validate_stocks(stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate stocks data before insertion"""
    validation_result = {
        "total_stocks": len(stocks_data),
        "valid_stocks": 0,
        "invalid_stocks": 0,
        "duplicate_symbols": [],
        "missing_data": [],
        "industries": set()
    }
    
    seen_symbols = set()
    
    for i, stock in enumerate(stocks_data):
        # Check for required fields
        if not stock.get('company_name') or not stock.get('symbol') or not stock.get('industry'):
            validation_result["missing_data"].append(f"Row {i+1}: Missing required data")
            validation_result["invalid_stocks"] += 1
            continue
        
        # Check for duplicate symbols
        symbol = stock['symbol']
        if symbol in seen_symbols:
            validation_result["duplicate_symbols"].append(symbol)
            validation_result["invalid_stocks"] += 1
        else:
            seen_symbols.add(symbol)
            validation_result["valid_stocks"] += 1
            validation_result["industries"].add(stock['industry'])
    
    validation_result["industries"] = sorted(list(validation_result["industries"]))
    return validation_result

def main():
    parser = argparse.ArgumentParser(description='Populate Nifty 500 stocks from CSV file')
    parser.add_argument('csv_file', help='Path to the Nifty 500 CSV file')
    parser.add_argument('--clear', action='store_true', help='Clear existing stocks before populating')
    parser.add_argument('--validate-only', action='store_true', help='Only validate data without inserting')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        logger.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Read CSV file
    logger.info(f"Reading CSV file: {args.csv_file}")
    stocks_data = read_nifty500_csv(args.csv_file)
    
    if not stocks_data:
        logger.error("No valid stocks found in CSV file")
        sys.exit(1)
    
    # Validate stocks
    logger.info("Validating stocks data...")
    validation = validate_stocks(stocks_data)
    
    logger.info(f"Validation Results:")
    logger.info(f"  Total stocks: {validation['total_stocks']}")
    logger.info(f"  Valid stocks: {validation['valid_stocks']}")
    logger.info(f"  Invalid stocks: {validation['invalid_stocks']}")
    logger.info(f"  Duplicate symbols: {len(validation['duplicate_symbols'])}")
    logger.info(f"  Missing data entries: {len(validation['missing_data'])}")
    logger.info(f"  Industries: {len(validation['industries'])}")
    
    if validation['duplicate_symbols']:
        logger.warning(f"Duplicate symbols: {validation['duplicate_symbols'][:10]}...")
    
    if validation['missing_data']:
        logger.warning(f"Missing data entries: {validation['missing_data'][:5]}...")
    
    if validation['invalid_stocks'] > 0:
        logger.warning(f"Found {validation['invalid_stocks']} invalid stocks. Proceeding with valid ones.")
    
    # Show industries
    logger.info(f"Industries found: {', '.join(validation['industries'][:10])}{'...' if len(validation['industries']) > 10 else ''}")
    
    if args.validate_only:
        logger.info("Validation complete. Exiting without inserting data.")
        sys.exit(0)
    
    if args.dry_run:
        logger.info("Dry run mode. Would insert the following stocks:")
        for i, stock in enumerate(stocks_data[:5]):  # Show first 5
            logger.info(f"  {i+1}. {stock['symbol']} - {stock['company_name']} ({stock['industry']})")
        if len(stocks_data) > 5:
            logger.info(f"  ... and {len(stocks_data) - 5} more stocks")
        sys.exit(0)
    
    # Populate stocks
    logger.info("Populating stocks in database...")
    result = populate_stocks(stocks_data, clear_existing=args.clear)
    
    if 'error' in result:
        logger.error(f"Failed to populate stocks: {result['error']}")
        sys.exit(1)
    
    logger.info(f"Successfully populated stocks:")
    logger.info(f"  Created: {result.get('created_count', 0)}")
    logger.info(f"  Failed: {result.get('failed_count', 0)}")
    
    if result.get('errors'):
        logger.warning(f"Errors: {result['errors']}")
    
    logger.info("Stock population completed successfully!")

if __name__ == "__main__":
    main()
