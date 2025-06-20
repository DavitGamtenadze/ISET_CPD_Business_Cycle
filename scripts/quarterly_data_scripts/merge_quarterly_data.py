#!/usr/bin/env python
"""
Quarterly Data Merger

Merges quarterly GDP data with monetary policy rates, exchange rates, and inflation data.
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
        logging.FileHandler(log_dir / 'quarterly_data_merger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_quarterly_data():
    """Load all quarterly datasets."""
    logger.info("Loading quarterly datasets...")
    
    datasets = {}
    
    try:
        # Load quarterly GDP data
        gdp_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_gdp_processed.xlsx'
        if gdp_path.exists():
            datasets['gdp'] = pd.read_excel(gdp_path)
            logger.info("Quarterly GDP data loaded")
        else:
            logger.warning("Quarterly GDP data not found")
            
        # Load quarterly monetary policy rates
        monetary_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_monetary_policy_rates.xlsx'
        if monetary_path.exists():
            datasets['monetary'] = pd.read_excel(monetary_path)
            logger.info("Quarterly monetary policy rates loaded")
        else:
            logger.warning("Quarterly monetary policy rates not found")
            
        # Load quarterly REER data
        reer_path = project_root / 'data' / 'quarterly_data' / 'georgia_quarterly_reer.xlsx'
        if reer_path.exists():
            datasets['reer'] = pd.read_excel(reer_path)
            logger.info("Quarterly REER data loaded")
        else:
            logger.warning("Quarterly REER data not found")
            
        # Load quarterly NEER data
        neer_path = project_root / 'data' / 'quarterly_data' / 'georgia_quarterly_neer.xlsx'
        if neer_path.exists():
            datasets['neer'] = pd.read_excel(neer_path)
            logger.info("Quarterly NEER data loaded")
        else:
            logger.warning("Quarterly NEER data not found")
            
        # Load quarterly inflation data
        inflation_path = project_root / 'data' / 'quarterly_data' / 'georgia_quarterly_inflation.xlsx'
        if inflation_path.exists():
            datasets['inflation'] = pd.read_excel(inflation_path)
            logger.info("Quarterly inflation data loaded")
        else:
            logger.warning("Quarterly inflation data not found")
            
    except Exception as e:
        logger.error(f"Error loading quarterly data: {e}")
        raise
        
    return datasets


def merge_quarterly_datasets(datasets):
    """Merge all quarterly datasets on Date column."""
    logger.info("Merging quarterly datasets...")
    
    if 'gdp' not in datasets:
        logger.error("GDP data is required for merging")
        raise ValueError("GDP data not found")
        
    # Start with GDP data as base
    merged_data = datasets['gdp'].copy()
    logger.info(f"Starting with GDP data: {merged_data.shape}")
    
    # Clean GDP data (remove year-only entries)
    merged_data = merged_data[merged_data['Date'].astype(str).str.len() > 4]
    
    # Merge monetary policy rates
    if 'monetary' in datasets:
        merged_data = pd.merge(
            merged_data, 
            datasets['monetary'], 
            on='Date', 
            how='left',
            suffixes=('', '_monetary')
        )
        logger.info(f"After monetary merge: {merged_data.shape}")
    
    # Merge REER data
    if 'reer' in datasets:
        reer_data = datasets['reer'].copy()
        # Rename REER columns to avoid conflicts
        reer_cols = [col for col in reer_data.columns if col != 'Date']
        reer_rename = {col: f'REER_{col}' for col in reer_cols}
        reer_data = reer_data.rename(columns=reer_rename)
        
        merged_data = pd.merge(
            merged_data, 
            reer_data, 
            on='Date', 
            how='left'
        )
        logger.info(f"After REER merge: {merged_data.shape}")
    
    # Merge NEER data
    if 'neer' in datasets:
        neer_data = datasets['neer'].copy()
        # Rename NEER columns to avoid conflicts
        neer_cols = [col for col in neer_data.columns if col != 'Date']
        neer_rename = {col: f'NEER_{col}' for col in neer_cols}
        neer_data = neer_data.rename(columns=neer_rename)
        
        merged_data = pd.merge(
            merged_data, 
            neer_data, 
            on='Date', 
            how='left'
        )
        logger.info(f"After NEER merge: {merged_data.shape}")
    
    # Merge inflation data
    if 'inflation' in datasets:
        inflation_data = datasets['inflation'].copy()
        # Rename inflation columns to avoid conflicts
        inflation_cols = [col for col in inflation_data.columns if col != 'Date']
        inflation_rename = {col: f'Quarterly_{col}' for col in inflation_cols}
        inflation_data = inflation_data.rename(columns=inflation_rename)
        
        merged_data = pd.merge(
            merged_data, 
            inflation_data, 
            on='Date', 
            how='left'
        )
        logger.info(f"After inflation merge: {merged_data.shape}")
    
    logger.info(f"Final merged quarterly data: {merged_data.shape}")
    logger.info(f"Columns: {list(merged_data.columns)}")
    
    return merged_data


def save_merged_quarterly_data(df):
    """Save the merged quarterly data."""
    try:
        output_dir = project_root / 'data' / 'quarterly_data'
        output_dir.mkdir(exist_ok=True)
        
        excel_path = output_dir / 'georgia_quarterly_merged_data.xlsx'
        
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Merged quarterly data saved to {excel_path}")
        
    except Exception as e:
        logger.error(f"Error saving merged quarterly data: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting quarterly data merger...")
    
    try:
        # Load all quarterly datasets
        datasets = load_quarterly_data()
        
        # Merge the datasets
        merged_data = merge_quarterly_datasets(datasets)
        
        # Save the merged data
        save_merged_quarterly_data(merged_data)
        
        logger.info("Quarterly data merger completed successfully!")
        
        # Display summary
        print("\n" + "="*60)
        print("QUARTERLY DATA MERGER SUMMARY")
        print("="*60)
        print(f"Total quarterly observations: {len(merged_data)}")
        if len(merged_data) > 0:
            print(f"Date range: {merged_data['Date'].iloc[0]} to {merged_data['Date'].iloc[-1]}")
        else:
            print("No data available - empty datasets")
        print(f"Total columns: {len(merged_data.columns)}")
        print("\nDatasets merged:")
        for key in datasets.keys():
            print(f"  - {key.upper()}")
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in quarterly data merger: {e}")
        raise


if __name__ == "__main__":
    main() 