#!/usr/bin/env python
"""
Quarterly GDP Data Processing

This script processes raw quarterly GDP data and prepares it for disaggregation.
"""

import pandas as pd
import os
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
        logging.FileHandler(log_dir / 'quarterly_gdp_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_gdp_data(input_path, output_path):
    """Clean and pivot Georgia GDP data from HTML/Excel.
    
    Takes the raw quarterly GDP data and:
    - Removes extra columns
    - Keeps quarter/year format (e.g. 'I 10', 'II 10')
    - Pivots dates into rows
    
    Args:
        input_path: Path to raw file (HTML or Excel)
        output_path: Where to save processed file
    """
    try:
        # First try to read as HTML (since the file is actually HTML)
        try:
            import lxml  # Required for HTML parsing
            tables = pd.read_html(input_path)
            if tables:
                gdp_data = tables[0]  # Take the first table
                logger.info(f"Loaded GDP data from HTML: {gdp_data.shape} rows/cols")
            else:
                raise ValueError("No tables found in HTML")
        except (ImportError, ValueError):
            # Fallback to Excel reading
            try:
                gdp_data = pd.read_excel(input_path, engine='openpyxl')
                logger.info(f"Loaded GDP data from Excel: {gdp_data.shape} rows/cols")
            except:
                # Try with xlrd for older Excel files
                gdp_data = pd.read_excel(input_path, engine='xlrd')
                logger.info(f"Loaded GDP data from Excel (xlrd): {gdp_data.shape} rows/cols")
        
        # Drop first col but keep date headers
        gdp_data = gdp_data.iloc[:, 1:]
        
        # Reset index before pivot
        gdp_data = gdp_data.reset_index(drop=True)
        
        # Pivot so dates are rows
        gdp_pivoted = gdp_data.transpose()

        # Fix column names
        gdp_pivoted.columns = gdp_pivoted.iloc[0]
        
        # Try different column name variations for the date column
        date_column_names = ['Economic Activities', 'Date', gdp_pivoted.columns[0]]
        for date_col in date_column_names:
            if date_col in gdp_pivoted.columns:
                gdp_pivoted = gdp_pivoted.rename(columns={date_col: 'Date'})
                break
        else:
            # If no standard date column found, use the first column as Date
            gdp_pivoted.columns = ['Date'] + list(gdp_pivoted.columns[1:])
        
        gdp_pivoted = gdp_pivoted.iloc[1:] # Drop first row
        
        # Save it
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.xlsx'):
            gdp_pivoted.to_excel(output_path, index=False)
            
        logger.info(f"Saved clean data to {output_path}")
        logger.info(f"Final shape: {gdp_pivoted.shape}")
        logger.debug(f"\nFirst few rows:\n{gdp_pivoted.head()}")
        
    except FileNotFoundError:
        logger.error(f"Can't find input file: {input_path}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)

if __name__ == '__main__':
    logger.info("Starting GDP processing")
    
    # File paths using project root
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data' / 'preliminary_data' / 'georgia_quarterly_gdp.xlsx'
    output_file = project_root / 'data' / 'processed_data' / 'georgia_quarterly_gdp_processed.xlsx'
    
    # Process the quarterly GDP data 
    try:
        process_gdp_data(str(input_file), str(output_file))
    except Exception as e:
        logger.error(f"Error processing quarterly GDP file: {e}")
        logger.info("The quarterly GDP file seems to be HTML or corrupted.")
        logger.info("Please check if you have the correct quarterly GDP data file.")
        # Don't create empty data - let the error propagate so user knows there's an issue
    
    logger.info("Done!") 