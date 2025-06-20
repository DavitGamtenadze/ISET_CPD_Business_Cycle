#!/usr/bin/env python
"""
Quarterly Exchange Rate Data Processing

Converts REER and NEER data from monthly to quarterly format.
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
        logging.FileHandler(log_dir / 'quarterly_exchange_rates_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_monthly_to_quarterly(df, data_type):
    """Convert monthly exchange rate data to quarterly format."""
    logger.info(f"Converting {data_type} data to quarterly format...")
    
    try:
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
        
        # Group by quarter and take the mean of the quarter
        quarterly_data = df_clean.groupby('Quarter_Label')[numeric_cols].mean().reset_index()
        quarterly_data = quarterly_data.rename(columns={'Quarter_Label': 'Date'})
        
        # Sort by the original order
        date_order = df_clean.groupby('Quarter_Label')['Date'].min().sort_values()
        quarterly_data['sort_date'] = quarterly_data['Date'].map(date_order)
        quarterly_data = quarterly_data.sort_values('sort_date').drop('sort_date', axis=1)
        
        logger.info(f"Converted {data_type}: {quarterly_data.shape}")
        logger.info(f"Date range: {quarterly_data['Date'].iloc[0]} to {quarterly_data['Date'].iloc[-1]}")
        
        return quarterly_data
        
    except Exception as e:
        logger.error(f"Error converting {data_type} to quarterly: {e}")
        raise


def process_reer_data():
    """Process Real Effective Exchange Rate data."""
    logger.info("Processing REER data...")
    
    try:
        input_path = project_root / 'data' / 'processed_data' / 'georgia_real_effective_exchange_rate_processed.xlsx'
        
        if not input_path.exists():
            logger.warning("REER processed file not found, skipping...")
            return None
            
        df = pd.read_excel(input_path)
        return convert_monthly_to_quarterly(df, "REER")
        
    except Exception as e:
        logger.error(f"Error processing REER data: {e}")
        return None


def process_neer_data():
    """Process Nominal Effective Exchange Rate data."""
    logger.info("Processing NEER data...")
    
    try:
        input_path = project_root / 'data' / 'processed_data' / 'georgia_nominal_effective_exchange_rate_processed.xlsx'
        
        if not input_path.exists():
            logger.warning("NEER processed file not found, skipping...")
            return None
            
        df = pd.read_excel(input_path)
        return convert_monthly_to_quarterly(df, "NEER")
        
    except Exception as e:
        logger.error(f"Error processing NEER data: {e}")
        return None


def save_quarterly_exchange_rates(reer_df, neer_df):
    """Save quarterly exchange rate data."""
    try:
        output_dir = project_root / 'data' / 'quarterly_data'
        output_dir.mkdir(exist_ok=True)
        
        if reer_df is not None:
            reer_excel = output_dir / 'georgia_quarterly_reer.xlsx'
            reer_csv = output_dir / 'georgia_quarterly_reer.csv'
            reer_df.to_excel(reer_excel, index=False)
            logger.info(f"Saved quarterly REER data to {reer_excel}")
        
        if neer_df is not None:
            neer_excel = output_dir / 'georgia_quarterly_neer.xlsx'
            neer_csv = output_dir / 'georgia_quarterly_neer.csv'
            neer_df.to_excel(neer_excel, index=False)
            logger.info(f"Saved quarterly NEER data to {neer_excel}")
            
    except Exception as e:
        logger.error(f"Error saving quarterly exchange rate data: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting quarterly exchange rate processing...")
    
    try:
        # Process both exchange rate types
        reer_quarterly = process_reer_data()
        neer_quarterly = process_neer_data()
        
        # Save the processed data
        save_quarterly_exchange_rates(reer_quarterly, neer_quarterly)
        
        logger.info("Quarterly exchange rate processing completed!")
        
    except Exception as e:
        logger.error(f"Error in quarterly exchange rate processing: {e}")
        raise


if __name__ == "__main__":
    main() 