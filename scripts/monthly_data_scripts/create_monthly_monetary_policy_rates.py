#!/usr/bin/env python
"""
Monthly Monetary Policy Rate Creation

Creates monthly monetary policy rates from quarterly data for monthly analysis.
"""

import pandas as pd
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
        logging.FileHandler(log_dir / 'monthly_monetary_policy_rates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_monthly_monetary_policy_rates():
    """Create monthly monetary policy rates from quarterly data."""
    logger.info("Creating monthly monetary policy rates...")
    
    try:
        # Load quarterly monetary policy rates
        quarterly_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_monetary_policy_rates.xlsx'
        if not quarterly_path.exists():
            logger.error("Quarterly monetary policy rates not found")
            return None
            
        quarterly_rates = pd.read_excel(quarterly_path)
        logger.info(f"Loaded {len(quarterly_rates)} quarterly rate observations")
        
        # Convert quarterly format to monthly
        monthly_rates = []
        
        for _, row in quarterly_rates.iterrows():
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
                    monthly_rates.append({
                        'Date': date,
                        'Monetary_Policy_Rate_Percent': row['Monetary_Policy_Rate_Percent'],
                        'Monetary_Policy_Rate_Decimal': row['Monetary_Policy_Rate_Decimal']
                    })
        
        monthly_df = pd.DataFrame(monthly_rates)
        monthly_df = monthly_df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Created {len(monthly_df)} monthly monetary policy rate observations")
        logger.info(f"Date range: {monthly_df['Date'].min()} to {monthly_df['Date'].max()}")
        
        return monthly_df
        
    except Exception as e:
        logger.error(f"Error creating monthly monetary policy rates: {e}")
        return None


def save_monthly_monetary_policy_rates(df):
    """Save monthly monetary policy rates."""
    try:
        if df is None:
            logger.warning("No monthly monetary policy rates to save")
            return
            
        output_dir = project_root / 'data' / 'processed_data'
        output_dir.mkdir(exist_ok=True)
        
        excel_path = output_dir / 'georgia_monthly_monetary_policy_rates.xlsx'
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Monthly monetary policy rates saved to {excel_path}")
        
    except Exception as e:
        logger.error(f"Error saving monthly monetary policy rates: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting monthly monetary policy rate creation...")
    
    try:
        # Create monthly rates
        monthly_rates = create_monthly_monetary_policy_rates()
        
        # Save the rates
        save_monthly_monetary_policy_rates(monthly_rates)
        
        logger.info("Monthly monetary policy rate creation completed!")
        
        if monthly_rates is not None:
            print("\n" + "="*60)
            print("MONTHLY MONETARY POLICY RATES SUMMARY")
            print("="*60)
            print(f"Total monthly observations: {len(monthly_rates)}")
            print(f"Date range: {monthly_rates['Date'].min()} to {monthly_rates['Date'].max()}")
            print(f"Rate range: {monthly_rates['Monetary_Policy_Rate_Percent'].min()}% to {monthly_rates['Monetary_Policy_Rate_Percent'].max()}%")
            print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in monthly monetary policy rate creation: {e}")
        raise


if __name__ == "__main__":
    main() 