#!/usr/bin/env python
"""
Quarterly Inflation Data Processing

Converts inflation data from monthly to quarterly format.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging to project logs directory
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'quarterly_inflation_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_inflation_to_quarterly():
    """Convert monthly inflation data to quarterly format."""
    logger.info("Converting inflation data to quarterly format...")
    
    try:
        # Load processed inflation data
        input_path = project_root / 'data' / 'processed_data' / 'georgia_monthly_inflation_processed.xlsx'
        
        if not input_path.exists():
            logger.warning("Inflation processed file not found, skipping...")
            return None
            
        df = pd.read_excel(input_path)
        logger.info(f"Loaded inflation data: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Remove rows with missing dates or year-only entries
        df_clean = df.dropna(subset=['Date'])
        df_clean = df_clean[~df_clean['Date'].astype(str).str.match(r'^\d{4}$')]
        
        # Convert date to datetime
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Create quarterly aggregation
        df_clean['Year'] = df_clean['Date'].dt.year
        df_clean['Quarter'] = df_clean['Date'].dt.quarter
        
        # Create quarter labels (I 10, II 10, etc.)
        quarter_map = {1: 'I', 2: 'II', 3: 'III', 4: 'IV'}
        df_clean['Quarter_Label'] = df_clean.apply(
            lambda x: f"{quarter_map[x['Quarter']]} {str(x['Year'])[2:]}", axis=1
        )
        
        # Select numeric columns for aggregation (exclude Date columns)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # For inflation, we take the mean of the quarter
        # This gives us the average quarterly inflation rate
        quarterly_inflation = df_clean.groupby('Quarter_Label')[numeric_cols].mean().reset_index()
        quarterly_inflation = quarterly_inflation.rename(columns={'Quarter_Label': 'Date'})
        
        # Sort by the original chronological order
        date_order = df_clean.groupby('Quarter_Label')['Date'].min().sort_values()
        quarterly_inflation['sort_date'] = quarterly_inflation['Date'].map(date_order)
        quarterly_inflation = quarterly_inflation.sort_values('sort_date').drop('sort_date', axis=1)
        
        logger.info(f"Converted inflation data: {quarterly_inflation.shape}")
        logger.info(f"Date range: {quarterly_inflation['Date'].iloc[0]} to {quarterly_inflation['Date'].iloc[-1]}")
        
        return quarterly_inflation
        
    except Exception as e:
        logger.error(f"Error converting inflation to quarterly: {e}")
        raise


def save_quarterly_inflation(df):
    """Save quarterly inflation data."""
    try:
        if df is None:
            logger.warning("No inflation data to save")
            return
            
        output_dir = project_root / 'data' / 'quarterly_data'
        output_dir.mkdir(exist_ok=True)
        
        excel_path = output_dir / 'georgia_quarterly_inflation.xlsx'
        csv_path = output_dir / 'georgia_quarterly_inflation.csv'
        
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Saved quarterly inflation data to {excel_path}")
        logger.info(f"Saved quarterly inflation data to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving quarterly inflation data: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting quarterly inflation processing...")
    
    try:
        # Process the inflation data
        quarterly_inflation = convert_inflation_to_quarterly()
        
        # Save the processed data
        save_quarterly_inflation(quarterly_inflation)
        
        logger.info("Quarterly inflation processing completed!")
        
    except Exception as e:
        logger.error(f"Error in quarterly inflation processing: {e}")
        raise


if __name__ == "__main__":
    main() 