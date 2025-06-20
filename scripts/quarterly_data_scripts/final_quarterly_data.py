#!/usr/bin/env python
"""
Final Quarterly Data Creation

Creates the final quarterly dataset with all requested columns from monthly data sources.
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
        logging.FileHandler(log_dir / 'final_quarterly_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_monthly_to_quarterly_format(df, date_col='Date'):
    """Convert monthly data to quarterly format with I 10, II 10, etc."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Create quarterly aggregation
    df['Year'] = df[date_col].dt.year
    df['Quarter'] = df[date_col].dt.quarter

    # Create quarter labels (I 10, II 10, etc.)
    quarter_map = {1: 'I', 2: 'II', 3: 'III', 4: 'IV'}
    df['Quarter_Label'] = df.apply(
        lambda x: f"{quarter_map[x['Quarter']]} {str(x['Year'])[2:]}", axis=1
    )

    # Select numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Group by quarter and take the mean
    quarterly_data = df.groupby('Quarter_Label')[numeric_cols].mean().reset_index()
    quarterly_data = quarterly_data.rename(columns={'Quarter_Label': 'Date'})

    # Sort by chronological order
    date_order = df.groupby('Quarter_Label')[date_col].min().sort_values()
    quarterly_data['sort_date'] = quarterly_data['Date'].map(date_order)
    quarterly_data = quarterly_data.sort_values('sort_date').drop('sort_date', axis=1)

    return quarterly_data


def create_monthly_monetary_policy_rates():
    """Create monthly monetary policy rates from quarterly data."""
    logger.info("Creating monthly monetary policy rates...")

    try:
        # Load quarterly monetary policy rates
        quarterly_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_monetary_policy_rates.xlsx'
        if not quarterly_path.exists():
            logger.warning("Quarterly monetary policy rates not found")
            return None

        quarterly_rates = pd.read_excel(quarterly_path)

        # Convert quarterly format to dates for interpolation
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
        return monthly_df

    except Exception as e:
        logger.error(f"Error creating monthly monetary policy rates: {e}")
        return None


def load_final_monthly_data():
    """Load all monthly datasets needed for final quarterly data."""
    logger.info("Loading final monthly datasets...")

    datasets = {}

    # Load Chow-Lin GDP data
    try:
        chow_lin_path = project_root / 'data' / 'processed_data' / 'georgia_monthly_gdp_chow_lin.xlsx'
        if chow_lin_path.exists():
            datasets['chow_lin'] = pd.read_excel(chow_lin_path)
            logger.info("Chow-Lin GDP data loaded")
        else:
            logger.warning("Chow-Lin GDP data not found")
    except Exception as e:
        logger.error(f"Error loading Chow-Lin data: {e}")

    # Load final monthly data (with exchange rates)
    try:
        final_monthly_path = project_root / 'data' / 'monthly_data' / 'final_gdp_exchange_rate_data.xlsx'
        if final_monthly_path.exists():
            datasets['final_monthly'] = pd.read_excel(final_monthly_path)
            logger.info("Final monthly data loaded")
        else:
            logger.warning("Final monthly data not found")
    except Exception as e:
        logger.error(f"Error loading final monthly data: {e}")

    # Load processed exchange rates (using correct file names)
    try:
        reer_path = project_root / 'data' / 'processed_data' / 'georgia_reer_data_cleaned_processed.xlsx'
        if reer_path.exists():
            datasets['reer'] = pd.read_excel(reer_path)
            logger.info("REER data loaded")
        else:
            logger.warning("REER data not found")
    except Exception as e:
        logger.error(f"Error loading REER data: {e}")

    try:
        neer_path = project_root / 'data' / 'processed_data' / 'georgia_neer_data_cleaned_processed.xlsx'
        if neer_path.exists():
            datasets['neer'] = pd.read_excel(neer_path)
            logger.info("NEER data loaded")
        else:
            logger.warning("NEER data not found")
    except Exception as e:
        logger.error(f"Error loading NEER data: {e}")

    # Create monthly monetary policy rates
    monetary_monthly = create_monthly_monetary_policy_rates()
    if monetary_monthly is not None:
        datasets['monetary_monthly'] = monetary_monthly
        logger.info("Monthly monetary policy rates created")

    return datasets


def load_business_confidence_data():
    """Load Georgia Business Confidence Index data."""
    logger.info("Loading Georgia Business Confidence Index data...")

    try:
        business_confidence_path = project_root / 'data' / 'preliminary_data' / 'georgia_business_confidence_index.xlsx'

        if not business_confidence_path.exists():
            logger.warning("Business confidence index data not found")
            return None

        business_data = pd.read_excel(business_confidence_path)

        # Convert date format to match quarterly format (e.g., "2019-Q4" -> "IV 19")
        def convert_date_format(date_str):
            year, quarter = date_str.split('-')
            quarter_num = quarter[1]  # Extract number from Q1, Q2, etc.
            quarter_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
            quarter_roman = quarter_map[quarter_num]
            year_short = year[2:]  # Get last 2 digits
            return f"{quarter_roman} {year_short}"

        business_data['Date'] = business_data['Date'].apply(convert_date_format)

        logger.info(f"Business confidence data: {len(business_data)} records")
        logger.info(f"Business confidence columns: {list(business_data.columns)}")

        return business_data

    except Exception as e:
        logger.error(f"Error loading business confidence data: {e}")
        return None


def create_final_quarterly_data(datasets):
    """Create final quarterly dataset with all requested columns."""
    logger.info("Creating final quarterly dataset...")

    base_data = None

    # Start with the most complete dataset
    if 'final_monthly' in datasets:
        base_data = datasets['final_monthly'].copy()
        logger.info(f"Using final_monthly as base: {base_data.shape}")
    elif 'chow_lin' in datasets:
        base_data = datasets['chow_lin'].copy()
        logger.info(f"Using chow_lin as base: {base_data.shape}")
    else:
        logger.error("No suitable base dataset found")
        return None

    # Merge monetary policy rates if available
    if 'monetary_monthly' in datasets:
        monetary_data = datasets['monetary_monthly']
        base_data = pd.merge(base_data, monetary_data, on='Date', how='left')
        logger.info(f"After monetary merge: {base_data.shape}")

    # Merge exchange rate data if needed
    if 'reer' in datasets and 'REER' not in str(base_data.columns):
        reer_data = datasets['reer'].copy()
        reer_data['Date'] = pd.to_datetime(reer_data['Date'])
        base_data = pd.merge(base_data, reer_data, on='Date', how='left')
        logger.info(f"After REER merge: {base_data.shape}")

    if 'neer' in datasets and 'NEER' not in str(base_data.columns):
        neer_data = datasets['neer'].copy()
        neer_data['Date'] = pd.to_datetime(neer_data['Date'])
        base_data = pd.merge(base_data, neer_data, on='Date', how='left')
        logger.info(f"After NEER merge: {base_data.shape}")

    # Convert to quarterly format
    quarterly_data = convert_monthly_to_quarterly_format(base_data)

    # Load and merge business confidence data
    business_confidence_data = load_business_confidence_data()
    if business_confidence_data is not None:
        quarterly_data = pd.merge(quarterly_data, business_confidence_data, on='Date', how='left')
        logger.info(f"After business confidence merge: {quarterly_data.shape}")
        logger.info("Business confidence data merged successfully")
    else:
        logger.warning("No business confidence data to merge")

    # Select and rename columns to match requirements
    desired_columns = [
        'Date',
        'Real_GDP_Growth',
        'Inflation_Rate',
        'GDP_at_market_prices_monthly',
        'GDP per capita in GEL_monthly',
        'GDP per capita, USD_monthly',
        'GDP in mil. USD_monthly',
        'Monetary_Policy_Rate_Percent',
        'Monetary_Policy_Rate_Decimal',
        'Business_Climate',
        'Business_expectation',
    ]

    # Add exchange rate columns that exist
    exchange_rate_patterns = [
        'NEER (Dec-2011=100)',
        'Real Effective Exchange Rate - Index (Dec-11=100)',
        'GEL/USD Real Exchange Rate - Index (Dec-11=100)',
        'GEL/EUR Real Exchange Rate - Index (Dec-11=100)',
        'GEL/TRY Real Exchange Rate - Index (Dec-11=100)',
        'GEL/RUB Real Exchange Rate - Index (Dec-11=100)'
    ]

    # Find existing columns that match patterns
    available_columns = []
    for col in quarterly_data.columns:
        if col in desired_columns:
            available_columns.append(col)
        else:
            for pattern in exchange_rate_patterns:
                if pattern in col or any(keyword in col for keyword in ['NEER', 'REER', 'Real Effective', 'GEL/USD', 'GEL/EUR', 'GEL/TRY', 'GEL/RUB']):
                    available_columns.append(col)
                    break

    # Select available columns
    final_columns = ['Date'] + [col for col in available_columns if col != 'Date']
    quarterly_final = quarterly_data[final_columns] if final_columns else quarterly_data

    logger.info(f"Final quarterly data: {quarterly_final.shape}")
    logger.info(f"Columns: {list(quarterly_final.columns)}")

    return quarterly_final


def save_final_quarterly_data(df):
    """Save final quarterly data."""
    try:
        output_dir = project_root / 'data' / 'quarterly_data'
        output_dir.mkdir(exist_ok=True)

        excel_path = output_dir / 'georgia_final_quarterly_data.xlsx'
        df.to_excel(excel_path, index=False)

        logger.info(f"Final quarterly data saved to {excel_path}")

    except Exception as e:
        logger.error(f"Error saving final quarterly data: {e}")
        raise


def main():
    """Main processing function."""
    logger.info("Starting final quarterly data creation...")

    try:
        # Load all monthly datasets
        datasets = load_final_monthly_data()

        if not datasets:
            logger.error("No datasets loaded")
            return

        # Create final quarterly data
        quarterly_data = create_final_quarterly_data(datasets)

        if quarterly_data is None:
            logger.error("Failed to create quarterly data")
            return

        # Save the data
        save_final_quarterly_data(quarterly_data)

        logger.info("Final quarterly data creation completed!")

        # Display summary
        print("\n" + "="*60)
        print("FINAL QUARTERLY DATA SUMMARY")
        print("="*60)
        print(f"Total quarterly observations: {len(quarterly_data)}")
        if len(quarterly_data) > 0:
            print(f"Date range: {quarterly_data['Date'].iloc[0]} to {quarterly_data['Date'].iloc[-1]}")
        print(f"Total columns: {len(quarterly_data.columns)}")
        print("\nColumns included:")
        for col in quarterly_data.columns:
            print(f"  - {col}")
        print("\n" + "="*60)

    except Exception as e:
        logger.error(f"Error in final quarterly data creation: {e}")
        raise


if __name__ == "__main__":
    main()