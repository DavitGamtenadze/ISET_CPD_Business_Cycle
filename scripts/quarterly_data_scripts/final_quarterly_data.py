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
    """Convert monthly data to quarterly format with I 10, II 10, etc.
    
    GDP columns are SUMMED (total quarterly output)
    Rates/inflation columns are AVERAGED (quarterly average rates)
    """
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

    # Define which columns should be summed vs averaged
    # GDP level columns (absolute values that should be summed)
    gdp_columns = [col for col in numeric_cols if any(keyword in col.lower() for keyword in 
                   ['gdp_at_market_prices', 'gdp per capita', 'gdp in mil'])]
    
    # Rate/growth/index columns (percentages/ratios that should be averaged)
    rate_columns = [col for col in numeric_cols if any(keyword in col.lower() for keyword in 
                    ['rate', 'inflation', 'growth', 'confidence', 'index', 'neer', 'reer', 'exchange']) 
                    and col not in gdp_columns]
    
    # Remaining numeric columns (default to mean)
    other_columns = [col for col in numeric_cols if col not in gdp_columns and col not in rate_columns]

    quarterly_data = df.groupby('Quarter_Label').agg({
        **{col: 'sum' for col in gdp_columns},      # GDP columns: SUM
        **{col: 'mean' for col in rate_columns},    # Rates: MEAN
        **{col: 'mean' for col in other_columns}    # Others: MEAN
    }).reset_index()
    
    quarterly_data = quarterly_data.rename(columns={'Quarter_Label': 'Date'})

    # Sort by chronological order
    date_order = df.groupby('Quarter_Label')[date_col].min().sort_values()
    quarterly_data['sort_date'] = quarterly_data['Date'].map(date_order)
    quarterly_data = quarterly_data.sort_values('sort_date').drop('sort_date', axis=1)

    logger.info(f"GDP columns (summed): {gdp_columns}")
    logger.info(f"Rate columns (averaged): {rate_columns}")

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


def load_actual_quarterly_gdp_data():
    """Load actual quarterly GDP data from georgia_quarterly_gdp.xlsx."""
    logger.info("Loading actual quarterly GDP data")

    try:
        project_root = Path(__file__).parent.parent.parent
        quarterly_gdp_path = project_root / 'data' / 'preliminary_data' / 'georgia_quarterly_gdp.xlsx'

        if not quarterly_gdp_path.exists():
            logger.warning("Quarterly GDP data file not found")
            return None

        # Read raw data without headers
        df_raw = pd.read_excel(quarterly_gdp_path, header=None)
        
        # Extract quarterly dates from row 1
        header_row = df_raw.iloc[1]
        quarterly_dates = []
        date_columns = []
        
        for i, val in enumerate(header_row):
            if isinstance(val, str) and any(q in val for q in ['I ', 'II ', 'III ', 'IV ']):
                quarterly_dates.append(val.strip())
                date_columns.append(i)
        
        # Find all GDP-related rows
        gdp_rows = {}
        gdp_row_mappings = {
            'market prices': 'GDP_at_market_prices_quarterly',
            'per capita in gel': 'GDP_per_capita_GEL_quarterly', 
            'per capita, usd': 'GDP_per_capita_USD_quarterly',
            'mil. usd': 'GDP_mil_USD_quarterly'
        }
        
        for i in range(2, df_raw.shape[0]):
            second_col = df_raw.iloc[i, 1]
            if pd.notna(second_col):
                second_str = str(second_col).lower()
                for key, col_name in gdp_row_mappings.items():
                    if key in second_str and ('output' in second_str or 'gdp' in second_str):
                        gdp_rows[col_name] = i
                        logger.info(f"Found {col_name} at row {i}: {second_col}")
        
        if not gdp_rows:
            logger.error("No GDP rows found in quarterly data")
            return None
        
        # Extract GDP values for all found columns
        gdp_data = []
        
        for i, col_idx in enumerate(date_columns):
            date = quarterly_dates[i]
            row_data = {'Date': date}
            
            # Extract values for each GDP column
            for col_name, row_idx in gdp_rows.items():
                gdp_row = df_raw.iloc[row_idx]
                value = gdp_row.iloc[col_idx]
                
                if pd.notna(value) and isinstance(value, (int, float)):
                    row_data[col_name] = float(value)
                else:
                    row_data[col_name] = None
            
            # Only add if we have at least the main GDP value
            if 'GDP_at_market_prices_quarterly' in row_data and row_data['GDP_at_market_prices_quarterly'] is not None:
                gdp_data.append(row_data)
        
        quarterly_gdp_df = pd.DataFrame(gdp_data)
        
        # Filter to start from I 12 as requested (2012, not 2011)
        start_idx = None
        for i, date in enumerate(quarterly_gdp_df['Date']):
            if date.strip() == 'I 12':  # Exact match for I 12 (2012)
                start_idx = i
                logger.info(f"Starting from I 12 at index {i}")
                break
        
        if start_idx is not None:
            quarterly_gdp_df = quarterly_gdp_df.iloc[start_idx:].reset_index(drop=True)
        
        logger.info(f"Actual quarterly GDP data: {len(quarterly_gdp_df)} records")
        logger.info(f"Date range: {quarterly_gdp_df['Date'].iloc[0]} to {quarterly_gdp_df['Date'].iloc[-1]}")
        
        return quarterly_gdp_df

    except Exception as e:
        logger.error(f"Error loading actual quarterly GDP data: {e}")
        return None


def load_weighted_deposit_rates_data():
    """Load Georgia weighted deposit interest rates data."""
    logger.info("Loading weighted deposit interest rates data")

    try:
        project_root = Path(__file__).parent.parent.parent
        deposit_rates_path = project_root / 'data' / 'preliminary_data' / 'georgia_quarterly_weighted_deposit_interest_rates.xlsx'

        if not deposit_rates_path.exists():
            logger.warning("Weighted deposit interest rates data not found")
            return None

        # Read data, skip first row, no header
        deposit_data = pd.read_excel(deposit_rates_path, header=None, skiprows=1)
        
        # Column 0 is dates, Column 4 is "სულ - ეროვნული ვალუტით"
        deposit_clean = deposit_data[[0, 4]].copy()
        deposit_clean.columns = ['Date', 'Weighted_Deposit_Rate_GEL']
        
        # Remove rows with NaN dates
        deposit_clean = deposit_clean[deposit_clean['Date'].notna()]
        
        # Convert date format from "1996-Q-1" to "I 96"
        def convert_deposit_date_format(date_str):
            if pd.isna(date_str) or not isinstance(date_str, str):
                return None
            
            try:
                # Handle format like "1996-Q-1"
                if '-Q-' in date_str:
                    year, quarter_part = date_str.split('-Q-')
                    quarter_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
                    quarter_roman = quarter_map.get(quarter_part)
                    if quarter_roman:
                        year_short = year[2:]  # Get last 2 digits
                        return f"{quarter_roman} {year_short}"
                return None
            except (ValueError, IndexError):
                return None
        
        deposit_clean['Date'] = deposit_clean['Date'].apply(convert_deposit_date_format)
        
        # Remove rows with invalid dates
        deposit_clean = deposit_clean[deposit_clean['Date'].notna()]
        
        # Convert deposit rates to numeric, handle non-numeric values
        deposit_clean['Weighted_Deposit_Rate_GEL'] = pd.to_numeric(deposit_clean['Weighted_Deposit_Rate_GEL'], errors='coerce')
        
        # Remove rows with invalid deposit rates
        deposit_clean = deposit_clean[deposit_clean['Weighted_Deposit_Rate_GEL'].notna()]
        
        # Divide by 100 to convert to decimal format (like other rates)
        deposit_clean['Weighted_Deposit_Rate_GEL'] = deposit_clean['Weighted_Deposit_Rate_GEL'] / 100.0
        
        logger.info(f"Weighted deposit rates data: {len(deposit_clean)} records")
        logger.info("Converted weighted deposit rates to decimal format")
        
        return deposit_clean

    except Exception as e:
        logger.error(f"Error loading weighted deposit rates data: {e}")
        return None


def load_business_confidence_data():
    """Load Georgia Business Confidence Index data."""
    logger.info("Loading Georgia Business Confidence Index data...")

    try:
        business_confidence_path = project_root / 'data' / 'preliminary_data' / 'georgia_business_confidence_index.xlsx'

        if not business_confidence_path.exists():
            logger.warning("Business confidence index data not found")
            return None

        business_data = pd.read_excel(business_confidence_path)

        # Convert date format to match quarterly format (e.g., "Q4/13" -> "IV 13")
        def convert_date_format(date_str):
            # Skip NaN or invalid values
            if pd.isna(date_str) or not isinstance(date_str, str):
                return None
                
            try:
                if '/' in date_str:
                    # Handle format like "Q4/13"
                    parts = date_str.split('/')
                    if len(parts) != 2:
                        return None
                    quarter, year = parts
                    quarter_num = quarter[1]  # Extract number from Q1, Q2, etc.
                    year_short = year  # Already short format
                elif '-' in date_str:
                    # Handle format like "2019-Q4"
                    parts = date_str.split('-')
                    if len(parts) != 2:
                        return None
                    year, quarter = parts
                    quarter_num = quarter[1]  # Extract number from Q1, Q2, etc.
                    year_short = year[2:]  # Get last 2 digits
                else:
                    return None
                
                quarter_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
                if quarter_num not in quarter_map:
                    return None
                quarter_roman = quarter_map[quarter_num]
                return f"{quarter_roman} {year_short}"
            except (IndexError, ValueError):
                return None

        business_data['Date'] = business_data['Date'].apply(convert_date_format)
        
        # Remove rows with invalid dates
        business_data = business_data[business_data['Date'].notna()]

        logger.info(f"Business confidence data: {len(business_data)} records")
        logger.info(f"Business confidence columns: {list(business_data.columns)}")

        return business_data

    except Exception as e:
        logger.error(f"Error loading business confidence data: {e}")
        return None


def create_final_quarterly_data(datasets):
    """Create final quarterly dataset with all requested columns."""
    logger.info("Creating final quarterly dataset...")

    # Load actual quarterly GDP data instead of using monthly aggregation
    quarterly_gdp_data = load_actual_quarterly_gdp_data()
    if quarterly_gdp_data is None:
        logger.error("Could not load actual quarterly GDP data")
        return None

    # Start with actual quarterly GDP data as base
    quarterly_data = quarterly_gdp_data.copy()
    logger.info(f"Using actual quarterly GDP as base: {quarterly_data.shape}")

    # Get monthly data for rates/indices aggregation (but not GDP)
    base_data = None
    if 'final_monthly' in datasets:
        base_data = datasets['final_monthly'].copy()
        logger.info(f"Monthly data available for rates: {base_data.shape}")
    elif 'chow_lin' in datasets:
        base_data = datasets['chow_lin'].copy()
        logger.info(f"Chow-Lin data available for rates: {base_data.shape}")

    if base_data is not None:
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

        # Convert monthly rates/indices to quarterly format (exclude GDP columns)
        rates_columns = [col for col in base_data.columns if col not in ['GDP_at_market_prices_monthly', 'GDP per capita in GEL_monthly', 'GDP per capita, USD_monthly', 'GDP in mil. USD_monthly']]
        rates_data = base_data[rates_columns]
        quarterly_rates = convert_monthly_to_quarterly_format(rates_data)
        
        # Merge quarterly rates with quarterly GDP
        quarterly_data = pd.merge(quarterly_data, quarterly_rates, on='Date', how='left')
        logger.info(f"After merging quarterly rates: {quarterly_data.shape}")
    else:
        logger.warning("No monthly data available for rates aggregation")

    # Load and merge business confidence data
    business_confidence_data = load_business_confidence_data()
    if business_confidence_data is not None:
        quarterly_data = pd.merge(quarterly_data, business_confidence_data, on='Date', how='left')
        
        logger.info(f"After business confidence merge: {quarterly_data.shape}")
        logger.info("Business confidence data merged successfully")
    else:
        logger.warning("No business confidence data to merge")

    # Load and merge weighted deposit interest rates data
    deposit_rates_data = load_weighted_deposit_rates_data()
    if deposit_rates_data is not None:
        quarterly_data = pd.merge(quarterly_data, deposit_rates_data, on='Date', how='left')
        logger.info(f"After deposit rates merge: {quarterly_data.shape}")
        logger.info("Weighted deposit rates data merged successfully")
    else:
        logger.warning("No weighted deposit rates data to merge")

    # Select and rename columns to match requirements
    desired_columns = [
        'Date',
        'GDP_at_market_prices_quarterly',  # Actual quarterly GDP data
        'GDP_per_capita_GEL_quarterly',    # GDP per capita in GEL
        'GDP_per_capita_USD_quarterly',    # GDP per capita in USD
        'GDP_mil_USD_quarterly',           # GDP in millions USD
        'Real_GDP_Growth',
        'Inflation_Rate',
        'Monetary_Policy_Rate_Decimal',    # Only decimal version, remove percent
        'Business Confidence Index (BCI)',
        'Sales Price Expectations Index',
        'Weighted_Deposit_Rate_GEL',
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