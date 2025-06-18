import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def process_gdp_data(input_path, output_path):
    """Clean and pivot Georgia GDP data from Excel.
    
    Takes the raw quarterly GDP data and:
    - Removes extra columns
    - Keeps quarter/year format (e.g. 'I 10', 'II 10')
    - Pivots dates into rows
    
    Args:
        input_path: Path to raw Excel file
        output_path: Where to save processed file
    """
    try:
        # Load data
        gdp_data = pd.read_excel(input_path)
        logger.info(f"Loaded GDP data: {gdp_data.shape} rows/cols")
        
        # Drop first col but keep date headers
        gdp_data = gdp_data.iloc[:, 1:]
        
        # Reset index before pivot
        gdp_data = gdp_data.reset_index(drop=True)
        
        # Pivot so dates are rows
        gdp_pivoted = gdp_data.transpose()

        # Fix column names
        gdp_pivoted.columns = gdp_pivoted.iloc[0]
        gdp_pivoted = gdp_pivoted.rename(columns={'Economic Activities': 'Date'})
        gdp_pivoted = gdp_pivoted.iloc[1:] # Drop first row
        
        # Save it
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.xlsx'):
            gdp_pivoted.to_excel(output_path, index=False)
        else:
            gdp_pivoted.to_csv(output_path, index=False)
            
        logger.info(f"Saved clean data to {output_path}")
        logger.debug(f"\nFirst few rows:\n{gdp_pivoted.head()}")
        
    except FileNotFoundError:
        logger.error(f"Can't find input file: {input_path}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)

if __name__ == '__main__':
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/gdp_processing.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting GDP processing")
    
    # File paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(root, 'data', 'preliminary_data', 'georgia_quarterly_gdp.xlsx')
    output_file = os.path.join(root, 'data', 'processed_data', 'georgia_quarterly_gdp_processed.xlsx')
    
    process_gdp_data(input_file, output_file)
    
    logger.info("Done!")
