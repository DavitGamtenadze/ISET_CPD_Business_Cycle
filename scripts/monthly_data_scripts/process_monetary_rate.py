#!/usr/bin/env python
"""
Monetary Policy Rate Data Processing

Processes Georgia's monetary policy rate data from monthly to quarterly format.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Setup logging to project logs directory
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'monetary_rate_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_monetary_policy_rates():
    """Process monetary policy rate data from monthly to quarterly."""
    logger.info("Processing monetary policy rate data...")
    
    try:
        # Load monetary policy rate data
        input_path = project_root / 'data' / 'preliminary_data' / 'georgia_monthly_monetary_policy_rate_changes.xlsx'
        df = pd.read_excel(input_path)
        
        # Skip first row (header in Georgian) as requested
        df = df.iloc[1:].reset_index(drop=True)
        logger.info(f"Loaded monetary rate data: {df.shape}")
        
        # The actual column headers are in the first row after skipping
        # Based on inspection: Column 0 = Decision Date, Column 1 = Implementation Date, Column 2 = Rate %
        df.columns = ['Decision_Date', 'Implementation_Date', 'Rate_Percent']
        
        logger.info(f"Columns renamed to: {df.columns.tolist()}")
        
        # Remove any rows with NaN in Implementation_Date
        df = df.dropna(subset=['Implementation_Date']).reset_index(drop=True)
        
        # Convert implementation date to datetime
        df['Implementation_Date'] = pd.to_datetime(df['Implementation_Date'])
        
        # Convert rate to numeric, handling any string values
        df['Rate_Percent'] = pd.to_numeric(df['Rate_Percent'], errors='coerce')
        
        # Remove rows where rate conversion failed
        df = df.dropna(subset=['Rate_Percent']).reset_index(drop=True)
        
        # Convert rate to decimal (from percentage points to decimal)
        # Value of 10 means 10%, so divide by 100 for calculations
        df['Rate_Decimal'] = df['Rate_Percent'] / 100
        
        # Create quarterly format
        df['Year'] = df['Implementation_Date'].dt.year
        df['Quarter'] = df['Implementation_Date'].dt.quarter
        
        # Remove rows with NaN quarters
        df = df.dropna(subset=['Year', 'Quarter']).reset_index(drop=True)
        
        # Create quarter labels (I 10, II 10, etc.)
        quarter_map = {1: 'I', 2: 'II', 3: 'III', 4: 'IV'}
        df['Quarter_Label'] = df.apply(lambda x: f"{quarter_map[int(x['Quarter'])]} {str(int(x['Year']))[2:]}", axis=1)
        
        # Group by quarter and take the last rate change in each quarter
        quarterly_rates = df.groupby('Quarter_Label').agg({
            'Rate_Percent': 'last',
            'Rate_Decimal': 'last',
            'Implementation_Date': 'last'
        }).reset_index()
        
        # Sort by implementation date to ensure proper order
        quarterly_rates = quarterly_rates.sort_values('Implementation_Date')
        
        # Rename columns for output
        quarterly_rates = quarterly_rates.rename(columns={
            'Quarter_Label': 'Date',
            'Rate_Percent': 'Monetary_Policy_Rate_Percent',
            'Rate_Decimal': 'Monetary_Policy_Rate_Decimal'
        })
        
        # Keep only necessary columns
        quarterly_rates = quarterly_rates[['Date', 'Monetary_Policy_Rate_Percent', 'Monetary_Policy_Rate_Decimal']]
        
        logger.info(f"Processed quarterly rates: {quarterly_rates.shape}")
        if len(quarterly_rates) > 0:
            logger.info(f"Date range: {quarterly_rates['Date'].iloc[0]} to {quarterly_rates['Date'].iloc[-1]}")
        
        return quarterly_rates
        
    except Exception as e:
        logger.error(f"Error processing monetary policy rates: {e}")
        raise


def save_quarterly_monetary_rates(df):
    """Save quarterly monetary policy rates."""
    try:
        # Save to processed data
        output_dir = project_root / 'data' / 'processed_data'
        output_dir.mkdir(exist_ok=True)
        
        excel_path = output_dir / 'georgia_quarterly_monetary_policy_rates.xlsx'
        
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Saved quarterly monetary rates to {excel_path}")
        
    except Exception as e:
        logger.error(f"Error saving quarterly monetary rates: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting monetary policy rate processing...")
    
    try:
        # Process the data
        quarterly_rates = process_monetary_policy_rates()
        
        # Save the processed data
        save_quarterly_monetary_rates(quarterly_rates)
        
        logger.info("Monetary policy rate processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in monetary policy rate processing: {e}")
        raise


if __name__ == "__main__":
    main() 