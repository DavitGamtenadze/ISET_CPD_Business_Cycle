"""
Chow-Lin GDP Disaggregation Script

This script converts quarterly GDP data to monthly estimates using the Chow-Lin method. -- 2012 and onwards
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from tempdisagg import TempDisaggModel

# Setup logging to project logs directory
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'chow_lin_gdp_disaggregation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChowLinGDPModel:
    """Chow-Lin GDP Disaggregation Model for quarterly to monthly conversion."""
    
    def __init__(self, method: str = 'chow-lin-opt', conversion: str = 'sum'):
        self.method = method
        self.conversion = conversion
        self.models = {}
        self.fitted = False
        
    def merge_quarterly_monthly_gdp(self, 
                                   inflation_data: pd.DataFrame,
                                   quarterly_gdp: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging quarterly and monthly GDP data...")
        
        quarterly_data = quarterly_gdp.copy()
        parts = quarterly_data['Date'].str.extract(r'^(I{1,3}|IV)\s+(\d{2})\*?$')
        quarterly_data['Q_Roman'], quarterly_data['YY2'] = parts[0], parts[1]

        quarterly_data = quarterly_data[quarterly_data['Q_Roman'].isin(['I','II','III','IV'])].copy()

        roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
        quarterly_data['qnum'] = quarterly_data['Q_Roman'].map(roman_map).astype(int)
        quarterly_data['year4'] = quarterly_data['YY2'].astype(int) + 2000
        quarterly_data['q_str'] = quarterly_data['year4'].astype(str) + 'Q' + quarterly_data['qnum'].astype(str)

        gdp_cols = [
            '(=) GDP at market prices',
            'GDP per capita in GEL', 
            'GDP per capita, USD',
            'GDP in mil. USD'
        ]
        q_merge = quarterly_data[['q_str'] + gdp_cols]

        monthly_data = inflation_data.copy()
        monthly_data['Date'] = pd.to_datetime(monthly_data['Date'], dayfirst=True)
        monthly_data['q_str'] = monthly_data['Date'].dt.to_period('Q').astype(str)

        merged = pd.merge(monthly_data, q_merge, on='q_str', how='left')
        merged = merged.rename(columns={
            '(=) GDP at market prices': 'GDP_at_market_prices'
        })

        logger.info(f"Successfully merged data with {len(merged)} monthly observations")
        return merged

    def disaggregate_quarterly_to_monthly(self, 
                                        df: pd.DataFrame, 
                                        target_column: str, 
                                        indicator_column: str = 'Real_GDP_Growth') -> np.ndarray:
        logger.info(f"Disaggregating {target_column} using Chow-Lin method...")
        
        quarterly_data = df.drop_duplicates('q_str').sort_values('q_str').reset_index(drop=True)
        quarterly_values = quarterly_data[target_column].values
        num_quarters = len(quarterly_values)

        group_ids = np.repeat(np.arange(num_quarters), 3)
        grain_identifiers = np.tile([1, 2, 3], num_quarters)

        repeated_quarterly_values = np.repeat(quarterly_values, 3)
        monthly_indicators = df[indicator_column].values

        long_format_df = pd.DataFrame({
            'Index': group_ids,
            'Grain': grain_identifiers,
            'y': repeated_quarterly_values,
            'X': monthly_indicators,
        })

        model = TempDisaggModel(method=self.method, conversion=self.conversion)
        model.fit(long_format_df)
        self.models[target_column] = model
        
        predictions = model.predict(full=False)
        logger.info(f"Successfully disaggregated {target_column}")
        
        return predictions

    def fit_and_predict(self, 
                       merged_df: pd.DataFrame, 
                       gdp_columns: Optional[list] = None,
                       indicator_column: str = 'Real_GDP_Growth') -> pd.DataFrame:
        """
        Fit Chow-Lin models for all specified GDP columns and generate predictions.

        Parameters:
        -----------
        merged_df : pd.DataFrame
            The merged DataFrame containing quarterly and monthly data
        gdp_columns : list, optional
            List of GDP columns to disaggregate. If None, uses default columns
        indicator_column : str, default 'Real_GDP_Growth'
            The indicator column to use for disaggregation

        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus monthly GDP estimates
        """
        if gdp_columns is None:
            gdp_columns = [
                'GDP_at_market_prices',
                'GDP per capita in GEL',
                'GDP per capita, USD', 
                'GDP in mil. USD'
            ]

        result_df = merged_df.copy()
        
        logger.info(f"Fitting Chow-Lin models for {len(gdp_columns)} GDP columns...")
        
        for col in gdp_columns:
            if col in merged_df.columns:
                monthly_estimates = self.disaggregate_quarterly_to_monthly(
                    merged_df, col, indicator_column
                )
                result_df[f'{col}_monthly'] = monthly_estimates
            else:
                logger.warning(f"Column {col} not found in data, skipping...")

        self.fitted = True
        logger.info("All Chow-Lin models fitted successfully")
        
        return result_df

    def save_models(self, filepath: Union[str, Path]) -> None:
        """
        Save all fitted models to a pickle file.

        Parameters:
        -----------
        filepath : str or Path
            Path where to save the models
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before saving")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'method': self.method,
            'conversion': self.conversion,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: Union[str, Path]) -> None:
        """
        Load fitted models from a pickle file.

        Parameters:
        -----------
        filepath : str or Path
            Path to the saved models file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.models = model_data['models']
        self.method = model_data['method']
        self.conversion = model_data['conversion']
        self.fitted = model_data['fitted']
        
        logger.info(f"Models loaded from {filepath}")


def load_data() -> tuple:
    logger.info("Loading data files...")
    
    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent
    
    try:
        yoy_path = project_root / 'data' / 'preliminary_data' / 'georgia_monthly_yoy_growth_rates_2011_2024.xlsx'
        yoy_growth_rates = pd.read_excel(yoy_path)
        logger.info("YoY growth rates loaded successfully")
    except FileNotFoundError:
        logger.error("YoY growth rates file not found")
        raise

    try:
        inflation_path = project_root / 'data' / 'preliminary_data' / 'georgia_monthly_inflation_2004_2025.xlsx'
        inflation_data = pd.read_excel(inflation_path)
        inflation_data['Date'] = pd.to_datetime(inflation_data['Date'])
        inflation_data['Date'] = inflation_data['Date'].dt.to_period('M').apply(lambda x: x.start_time)
        logger.info("Inflation data loaded successfully")
    except FileNotFoundError:
        logger.error("Inflation data file not found")
        raise

    try:
        quarterly_path = project_root / 'data' / 'processed_data' / 'georgia_quarterly_gdp_processed.xlsx'
        quarterly_gdp = pd.read_excel(quarterly_path)
        logger.info(f"Data columns: {quarterly_gdp.columns}")
        quarterly_gdp = quarterly_gdp[quarterly_gdp['Date'].astype(str).str.len() > 4]
        quarterly_gdp = quarterly_gdp.iloc[:-1]
        logger.info("Quarterly GDP data loaded successfully")
    except FileNotFoundError:
        logger.error("Quarterly GDP data file not found")
        raise

    return yoy_growth_rates, inflation_data, quarterly_gdp


def prepare_merged_data(yoy_growth_rates: pd.DataFrame, 
                       inflation_data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preparing merged data...")
    
    inflation_data['Date'] = pd.to_datetime(inflation_data['Date'])
    yoy_growth_rates['Date'] = pd.to_datetime(yoy_growth_rates['Date'])

    inflation_subset = inflation_data[['Date', 'Total_CPI']].rename(columns={'Total_CPI': 'Inflation_Rate'})
    merged_data = pd.merge(yoy_growth_rates, inflation_subset, on='Date', how='left')

    merged_data['Real_GDP_Growth'] = (
        (1 + merged_data['YoY Growth (%)']/100)
        / (1 + merged_data['Inflation_Rate']/100)
        - 1
    ) * 100

    logger.info("Merged data prepared successfully")
    return merged_data


def main():
    logger.info("Starting Chow-Lin GDP disaggregation process...")
    
    try:
        yoy_growth_rates, inflation_data, quarterly_gdp = load_data()
        merged_data = prepare_merged_data(yoy_growth_rates, inflation_data)
        
        quarterly_gdp = quarterly_gdp[["Date", "(=) GDP at market prices", 
                                     "GDP per capita in GEL", "GDP per capita, USD", "GDP in mil. USD"]]
        
        merged_data = merged_data[:-3]
        quarterly_gdp = quarterly_gdp[4:]
        
        chow_lin_model = ChowLinGDPModel(method='chow-lin-opt', conversion='sum')
        merged_df = chow_lin_model.merge_quarterly_monthly_gdp(merged_data, quarterly_gdp)
        result_df = chow_lin_model.fit_and_predict(merged_df)
        
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / 'models'
        models_dir.mkdir(exist_ok=True)
        chow_lin_model.save_models(models_dir / 'chow_lin_gdp_models.pkl')
        
        output_dir = project_root / 'data' / 'processed_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(output_dir / 'georgia_monthly_gdp_chow_lin.xlsx', index=False)
        result_df.to_csv(output_dir / 'georgia_monthly_gdp_chow_lin.csv', index=False)
        
        logger.info("Chow-Lin GDP disaggregation completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Models saved to {models_dir}")
        
        # Display summary statistics
        print("\n" + "="*60)
        print("CHOW-LIN GDP DISAGGREGATION SUMMARY")
        print("="*60)
        print(f"Total monthly observations: {len(result_df)}")
        print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
        print(f"GDP columns disaggregated: {len([col for col in result_df.columns if col.endswith('_monthly')])}")
        print("\nMonthly GDP columns created:")
        for col in result_df.columns:
            if col.endswith('_monthly'):
                print(f"  - {col}")
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main() 