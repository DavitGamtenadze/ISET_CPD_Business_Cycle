import pandas as pd
import logging
from pathlib import Path

# Setup logging to project logs directory
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'merge_gdp_final_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_exchange_rate_data():
    """Load NEER and REER exchange rate data."""
    logger.info("Loading exchange rate data...")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        neer_path = project_root / 'data' / 'processed_data' / 'georgia_neer_data_cleaned_processed.xlsx'
        reer_path = project_root / 'data' / 'processed_data' / 'georgia_reer_data_cleaned_processed.xlsx'
        
        neer_data = pd.read_excel(neer_path)
        reer_data = pd.read_excel(reer_path)
        
        # Convert dates to proper format
        neer_data['Date'] = pd.to_datetime(neer_data['Date'])
        reer_data['Date'] = pd.to_datetime(reer_data['Date'])
        
        logger.info(f"NEER data: {len(neer_data)} records")
        logger.info(f"REER data: {len(reer_data)} records")
        
        return neer_data, reer_data
        
    except FileNotFoundError as e:
        logger.error(f"Exchange rate data file not found: {e}")
        raise


def load_monetary_policy_data():
    """Load monthly monetary policy rate data."""
    logger.info("Loading monthly monetary policy rate data...")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        monetary_path = project_root / 'data' / 'processed_data' / 'georgia_monthly_monetary_policy_rates.xlsx'
        
        if not monetary_path.exists():
            logger.warning("Monthly monetary policy rates file not found, attempting to create from quarterly data...")
            # Try to load quarterly data and convert
            quarterly_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_monetary_policy_rates.xlsx'
            if quarterly_path.exists():
                quarterly_rates = pd.read_excel(quarterly_path)
                monthly_rates = convert_quarterly_to_monthly_monetary(quarterly_rates)
                if monthly_rates is not None:
                    # Save for future use
                    monthly_rates.to_excel(monetary_path, index=False)
                    logger.info("Created monthly monetary policy rates from quarterly data")
                    return monthly_rates
            
            logger.warning("No monetary policy data available")
            return None
        
        monetary_data = pd.read_excel(monetary_path)
        monetary_data['Date'] = pd.to_datetime(monetary_data['Date'])
        
        # Divide monetary policy rate by 100 to make it similar to inflation (decimal format)
        if 'Monetary_Policy_Rate_Percent' in monetary_data.columns:
            monetary_data['Monetary_Policy_Rate'] = monetary_data['Monetary_Policy_Rate_Percent'] / 100.0
        elif 'Monetary_Policy_Rate_Decimal' in monetary_data.columns:
            monetary_data['Monetary_Policy_Rate'] = monetary_data['Monetary_Policy_Rate_Decimal']
        
        # Keep only necessary columns
        monetary_data = monetary_data[['Date', 'Monetary_Policy_Rate']]
        
        logger.info(f"Monetary policy data: {len(monetary_data)} records")
        return monetary_data
        
    except Exception as e:
        logger.error(f"Error loading monetary policy data: {e}")
        return None


def convert_quarterly_to_monthly_monetary(quarterly_df):
    """Convert quarterly monetary policy rates to monthly format."""
    try:
        monthly_rates = []
        
        for _, row in quarterly_df.iterrows():
            date_str = row['Date']  # e.g., "I 08"
            quarter_roman, year_short = date_str.split()
            
            quarter_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
            quarter = quarter_map[quarter_roman]
            year = 2000 + int(year_short)
            
            # Create monthly rates for each quarter (3 months)
            for month_offset in range(3):
                month = (quarter - 1) * 3 + month_offset + 1
                if month <= 12:
                    date = pd.Timestamp(year=year, month=month, day=1)
                    rate_percent = row.get('Monetary_Policy_Rate_Percent', row.get('Monetary_Policy_Rate', 0))
                    monthly_rates.append({
                        'Date': date,
                        'Monetary_Policy_Rate': rate_percent / 100.0  # Convert to decimal
                    })
        
        monthly_df = pd.DataFrame(monthly_rates)
        monthly_df = monthly_df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Converted {len(monthly_df)} monthly monetary policy rate observations")
        return monthly_df
        
    except Exception as e:
        logger.error(f"Error converting quarterly to monthly monetary data: {e}")
        return None


def load_gdp_data():
    """Load the Chow-Lin disaggregated GDP data."""
    logger.info("Loading Chow-Lin GDP data...")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        gdp_path = project_root / 'data' / 'processed_data' / 'georgia_monthly_gdp_chow_lin.xlsx'
        gdp_data = pd.read_excel(gdp_path)
        logger.info(f"GDP data: {len(gdp_data)} records")
        return gdp_data
        
    except FileNotFoundError as e:
        logger.error(f"GDP data file not found: {e}")
        raise


def merge_data(gdp_data, neer_data, reer_data, monetary_data=None):
    """Merge GDP data with exchange rate data and monetary policy data."""
    logger.info("Merging GDP data with exchange rate and monetary policy data...")
    
    # Merge NEER data with GDP data
    merged_data = gdp_data.merge(neer_data, on='Date', how='left')
    logger.info(f"After NEER merge: {len(merged_data)} records")
    
    # Merge REER data with the result
    merged_data = merged_data.merge(reer_data, on='Date', how='left')
    logger.info(f"After REER merge: {len(merged_data)} records")
    
    # Merge monetary policy data if available
    if monetary_data is not None:
        merged_data = merged_data.merge(monetary_data, on='Date', how='left')
        logger.info(f"After monetary policy merge: {len(merged_data)} records")
    else:
        logger.warning("No monetary policy data to merge")
    
    return merged_data


def clean_data(merged_data):
    """Remove unnecessary columns from the merged dataset."""
    logger.info("Cleaning data - removing unnecessary columns...")
    
    columns_to_drop = [
        'q_str', 
        'GDP_at_market_prices', 
        'GDP per capita in GEL', 
        'GDP per capita, USD', 
        'GDP in mil. USD'
    ]
    
    # Check which columns exist before dropping
    existing_columns = [col for col in columns_to_drop if col in merged_data.columns]
    missing_columns = [col for col in columns_to_drop if col not in merged_data.columns]
    
    if missing_columns:
        logger.warning(f"Columns not found in data: {missing_columns}")
    
    if existing_columns:
        cleaned_data = merged_data.drop(existing_columns, axis=1)
        logger.info(f"Dropped columns: {existing_columns}")
    else:
        cleaned_data = merged_data
        logger.warning("No columns were dropped")
    
    logger.info(f"Final dataset shape: {cleaned_data.shape}")
    logger.info(f"Final columns: {list(cleaned_data.columns)}")
    
    return cleaned_data


def save_data(data, output_path):
    """Save the final dataset."""
    logger.info(f"Saving final dataset to {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Excel only
    excel_path = output_path.with_suffix('.xlsx')
    data.to_excel(excel_path, index=False)
    logger.info(f"Excel file saved: {excel_path}")
    
    return excel_path


def display_summary(data):
    """Display summary information about the final dataset."""
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {len(data)}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Total columns: {len(data.columns)}")
    
    print("\nColumns in final dataset:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\nMissing values per column:")
    missing_counts = data.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    if missing_counts.sum() == 0:
        print("  No missing values found")
    
    print("\n" + "="*60)


def main():
    """Main function to process and merge GDP and exchange rate data."""
    logger.info("Starting GDP and exchange rate data merger...")
    
    try:
        # Load data
        neer_data, reer_data = load_exchange_rate_data()
        gdp_data = load_gdp_data()
        monetary_data = load_monetary_policy_data()
        
        # Merge data
        merged_data = merge_data(gdp_data, neer_data, reer_data, monetary_data)
        
        # Clean data
        final_data = clean_data(merged_data)
        
        # Save data
        project_root = Path(__file__).parent.parent.parent
        output_path = project_root / 'data' / 'processed_data' / 'final_gdp_data_for_monthly.xlsx'
        excel_path = save_data(final_data, output_path)
        
        # Display summary
        display_summary(final_data)
        
        logger.info("GDP and exchange rate data merger completed successfully!")
        print(f"\nFinal file saved:")
        print(f"  Excel: {excel_path}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main() 