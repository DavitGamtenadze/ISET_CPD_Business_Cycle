import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
from tempdisagg import TempDisaggModel

# Setup logging to project logs directory
project_root = Path(__file__).parent.parent.parent
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'chow_lin_model_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChowLinModelPredictor:
    
    def __init__(self, models_filepath: Union[str, Path]):
        self.models_filepath = Path(models_filepath)
        self.models = {}
        self.method = None
        self.conversion = None
        self.fitted = False
        self.load_models()
    
    def load_models(self) -> None:
        if not self.models_filepath.exists():
            raise FileNotFoundError(f"Model file not found: {self.models_filepath}")
            
        with open(self.models_filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.models = model_data['models']
        self.method = model_data['method']
        self.conversion = model_data['conversion']
        self.fitted = model_data['fitted']
        
        logger.info(f"Loaded {len(self.models)} Chow-Lin models from {self.models_filepath}")
    
    def get_available_models(self) -> List[str]:
        return list(self.models.keys())
    
    def predict_single_column(self, 
                            quarterly_data: np.ndarray,
                            monthly_indicators: np.ndarray,
                            model_name: str) -> np.ndarray:
        """
        Make predictions for a single GDP column.
        
        Parameters:
        -----------
        quarterly_data : np.ndarray
            Quarterly GDP values
        monthly_indicators : np.ndarray  
            Monthly indicator values
        model_name : str
            Name of the model to use (must be in available models)
            
        Returns:
        --------
        np.ndarray
            Monthly predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        num_quarters = len(quarterly_data)
        
        # Create group IDs and grain identifiers
        group_ids = np.repeat(np.arange(num_quarters), 3)
        grain_identifiers = np.tile([1, 2, 3], num_quarters)
        
        # Repeat quarterly values
        repeated_quarterly_values = np.repeat(quarterly_data, 3)
        
        # Assemble the long-format DataFrame
        long_format_df = pd.DataFrame({
            'Index': group_ids,
            'Grain': grain_identifiers,
            'y': repeated_quarterly_values,
            'X': monthly_indicators,
        })
        
        # Make predictions using the saved model
        predictions = model.predict(full=False)
        
        logger.info(f"Generated predictions for {model_name}")
        return predictions
    
    def predict_multiple_columns(self,
                                quarterly_data_dict: Dict[str, np.ndarray],
                                monthly_indicators: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions for multiple GDP columns.
        
        Parameters:
        -----------
        quarterly_data_dict : Dict[str, np.ndarray]
            Dictionary mapping column names to quarterly data
        monthly_indicators : np.ndarray
            Monthly indicator values
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary mapping column names to monthly predictions
        """
        predictions = {}
        
        for col_name, quarterly_data in quarterly_data_dict.items():
            if col_name in self.models:
                predictions[col_name] = self.predict_single_column(
                    quarterly_data, monthly_indicators, col_name
                )
            else:
                logger.warning(f"Model for '{col_name}' not found, skipping...")
        
        logger.info(f"Generated predictions for {len(predictions)} columns")
        return predictions
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models.
        
        Returns:
        --------
        Dict
            Dictionary containing model information
        """
        return {
            'method': self.method,
            'conversion': self.conversion,
            'fitted': self.fitted,
            'num_models': len(self.models),
            'available_models': list(self.models.keys()),
            'models_filepath': str(self.models_filepath)
        }


def load_example_data() -> tuple:
    logger.info("Loading example data for testing...")
    
    try:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'data' / 'processed_data' / 'georgia_monthly_gdp_chow_lin.xlsx'
        if data_path.exists():
            df = pd.read_excel(data_path)
            
            quarterly_gdp_cols = [
                'GDP_at_market_prices',
                'GDP per capita in GEL',
                'GDP per capita, USD',
                'GDP in mil. USD'
            ]
            
            quarterly_data = {}
            for col in quarterly_gdp_cols:
                if col in df.columns:
                    unique_quarterly = df.drop_duplicates('q_str').sort_values('q_str')[col].values
                    quarterly_data[col] = unique_quarterly
            
            monthly_indicators = df['Real_GDP_Growth'].values
            
            logger.info("Example data loaded successfully")
            return quarterly_data, monthly_indicators
        else:
            logger.warning("No processed data found for examples")
            return {}, np.array([])
            
    except Exception as e:
        logger.error(f"Error loading example data: {str(e)}")
        return {}, np.array([])


def demonstrate_model_usage():
    print("\n" + "="*60)
    print("CHOW-LIN MODEL DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize predictor
        project_root = Path(__file__).parent.parent.parent
        models_path = project_root / 'models' / 'chow_lin_gdp_models.pkl'
        if not models_path.exists():
            print(f"Models file not found at {models_path}")
            print("Please run chow_lin_gdp_disaggregation.py first to generate models.")
            return
        
        predictor = ChowLinModelPredictor(models_path)
        
        # Display model information
        model_info = predictor.get_model_info()
        print(f"Method: {model_info['method']}")
        print(f"Conversion: {model_info['conversion']}")
        print(f"Number of models: {model_info['num_models']}")
        print(f"Available models: {model_info['available_models']}")
        
        # Load example data
        quarterly_data, monthly_indicators = load_example_data()
        
        if len(quarterly_data) > 0 and len(monthly_indicators) > 0:
            # Make predictions for all available models
            predictions = predictor.predict_multiple_columns(quarterly_data, monthly_indicators)
            
            print(f"\nPredictions generated for {len(predictions)} GDP columns:")
            for col_name, pred_values in predictions.items():
                print(f"  - {col_name}: {len(pred_values)} monthly values")
                print(f"    Sample values: {pred_values[:5]}")  # Show first 5 values
        else:
            print("\nNo example data available for demonstration")
            
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
    
    print("\n" + "="*60)


def create_new_predictions(quarterly_gdp_file: str, 
                          monthly_indicators_file: str,
                          output_file: str):
    """
    Create new predictions using saved models with new data.
    
    Parameters:
    -----------
    quarterly_gdp_file : str
        Path to Excel file with quarterly GDP data
    monthly_indicators_file : str
        Path to Excel file with monthly indicators
    output_file : str
        Path to save the predictions
    """
    logger.info("Creating new predictions with user-provided data...")
    
    try:
        # Load the predictor
        project_root = Path(__file__).parent.parent.parent
        models_path = project_root / 'models' / 'chow_lin_gdp_models.pkl'
        predictor = ChowLinModelPredictor(models_path)
        
        # Load user data
        quarterly_df = pd.read_excel(quarterly_gdp_file)
        monthly_df = pd.read_excel(monthly_indicators_file)
        
        # Extract quarterly data for available models
        quarterly_data = {}
        available_models = predictor.get_available_models()
        
        for model_name in available_models:
            if model_name in quarterly_df.columns:
                quarterly_data[model_name] = quarterly_df[model_name].values
        
        # Extract monthly indicators (assuming 'Real_GDP_Growth' column)
        if 'Real_GDP_Growth' in monthly_df.columns:
            monthly_indicators = monthly_df['Real_GDP_Growth'].values
        else:
            raise ValueError("'Real_GDP_Growth' column not found in monthly indicators file")
        
        # Generate predictions
        predictions = predictor.predict_multiple_columns(quarterly_data, monthly_indicators)
        
        # Create output DataFrame
        result_df = monthly_df.copy()
        for col_name, pred_values in predictions.items():
            result_df[f'{col_name}_monthly'] = pred_values
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.xlsx':
            result_df.to_excel(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        print(f"Successfully created predictions and saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating new predictions: {str(e)}")
        raise


if __name__ == "__main__":
    # Run demonstration
    demonstrate_model_usage() 