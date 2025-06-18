#!/usr/bin/env python
"""
Quarterly Data Analysis

This script performs analysis on quarterly data and aggregates monthly data back to quarterly
for comparison and validation purposes.
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'quarterly_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_monthly_data():
    """Load processed monthly data."""
    logger.info("Loading monthly data for quarterly aggregation...")
    
    try:
        # Use absolute paths from project root
        project_root = Path(__file__).parent.parent.parent
        monthly_path = project_root / 'data' / 'monthly_data'
        
        # Look for final monthly data
        possible_files = [
            'final_gdp_exchange_rate_data.xlsx',
            'georgia_monthly_gdp_chow_lin.xlsx'
        ]
        
        for filename in possible_files:
            file_path = monthly_path / filename
            if file_path.exists():
                logger.info(f"Found monthly data file: {filename}")
                df = pd.read_excel(file_path)
                return df
        
        # Also check processed_data folder
        processed_path = project_root / 'data' / 'processed_data'
        for filename in possible_files:
            file_path = processed_path / filename
            if file_path.exists():
                logger.info(f"Found monthly data file in processed_data: {filename}")
                df = pd.read_excel(file_path)
                return df
        
        logger.warning("No monthly data file found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading monthly data: {e}")
        return None


def aggregate_monthly_to_quarterly(df):
    """Aggregate monthly data to quarterly for validation."""
    logger.info("Aggregating monthly data to quarterly...")
    
    if df is None or 'Date' not in df.columns:
        logger.error("Invalid data for aggregation")
        return None
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create quarter column
    df['Quarter'] = df['Date'].dt.to_period('Q')
    
    # Define aggregation rules
    agg_rules = {}
    
    # GDP columns to sum
    gdp_columns = [col for col in df.columns if 'GDP' in col and 'monthly' in col]
    for col in gdp_columns:
        agg_rules[col] = 'sum'
    
    # Growth rates and percentages to average
    rate_columns = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'rate', '%', 'inflation'])]
    for col in rate_columns:
        agg_rules[col] = 'mean'
    
    # Exchange rate indices to average
    exchange_columns = [col for col in df.columns if any(x in col.upper() for x in ['NEER', 'REER'])]
    for col in exchange_columns:
        agg_rules[col] = 'mean'
    
    if not agg_rules:
        logger.warning("No columns found for aggregation")
        return None
    
    # Perform aggregation
    quarterly_agg = df.groupby('Quarter').agg(agg_rules).reset_index()
    
    logger.info(f"Aggregated to {len(quarterly_agg)} quarters")
    logger.info(f"Aggregated columns: {list(agg_rules.keys())}")
    
    return quarterly_agg


def calculate_quarterly_statistics(df):
    """Calculate quarterly statistics and trends."""
    logger.info("Calculating quarterly statistics...")
    
    if df is None:
        return None
    
    stats = {}
    
    # GDP growth statistics
    gdp_columns = [col for col in df.columns if 'GDP' in col and 'monthly' in col]
    if gdp_columns:
        main_gdp = gdp_columns[0]  # Use first GDP column as main
        
        # Calculate quarter-over-quarter growth
        df[f'{main_gdp}_QoQ_growth'] = df[main_gdp].pct_change() * 100
        
        # Calculate year-over-year growth
        df[f'{main_gdp}_YoY_growth'] = df[main_gdp].pct_change(periods=4) * 100
        
        stats['avg_quarterly_growth'] = df[f'{main_gdp}_QoQ_growth'].mean()
        stats['avg_annual_growth'] = df[f'{main_gdp}_YoY_growth'].mean()
        stats['quarterly_volatility'] = df[f'{main_gdp}_QoQ_growth'].std()
    
    # Exchange rate statistics
    exchange_columns = [col for col in df.columns if any(x in col.upper() for x in ['NEER', 'REER'])]
    for col in exchange_columns:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
            stats[f'{col}_trend'] = np.polyfit(range(len(df)), df[col].fillna(df[col].mean()), 1)[0]
    
    logger.info("Quarterly statistics calculated")
    return df, stats


def save_quarterly_analysis(df, stats, output_path):
    """Save quarterly analysis results."""
    logger.info(f"Saving quarterly analysis to {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if df is not None:
        # Save quarterly data
        excel_path = output_path.with_suffix('.xlsx')
        df.to_excel(excel_path, index=False)
        logger.info(f"Quarterly data saved: {excel_path}")
        
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Quarterly data saved: {csv_path}")
    
    if stats:
        # Save statistics
        stats_path = output_path.parent / f"{output_path.stem}_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("QUARTERLY DATA STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.4f}\n")
        logger.info(f"Statistics saved: {stats_path}")


def main():
    """Main analysis function."""
    logger.info("Starting quarterly data analysis...")
    
    try:
        # Load monthly data
        monthly_data = load_monthly_data()
        
        if monthly_data is not None:
            # Aggregate to quarterly
            quarterly_data = aggregate_monthly_to_quarterly(monthly_data)
            
            if quarterly_data is not None:
                # Calculate statistics
                analyzed_data, stats = calculate_quarterly_statistics(quarterly_data)
                
                # Save results
                project_root = Path(__file__).parent.parent.parent
                output_path = project_root / 'data' / 'quarterly_data' / 'quarterly_analysis_results'
                save_quarterly_analysis(analyzed_data, stats, output_path)
                
                # Display summary
                print("\n" + "="*50)
                print("QUARTERLY ANALYSIS SUMMARY")
                print("="*50)
                if stats:
                    for key, value in stats.items():
                        print(f"{key}: {value:.4f}")
                print("="*50)
                
                logger.info("Quarterly analysis completed successfully!")
            else:
                logger.warning("No data to analyze")
        else:
            logger.warning("No monthly data found for analysis")
            
    except Exception as e:
        logger.error(f"Error in quarterly analysis: {e}")
        raise


if __name__ == "__main__":
    main() 