import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def process_neer_data(input_path, output_path):
    """
    Processes Nominal Effective Exchange Rate data from Excel.
    Specifically keeps only the "NEER (Dec-1995=100)" column and removes the first 49 rows.
    The output will have 12-1999 rebased to 100, with all other values as percentages relative to that.
    """
    logger.info(f"Processing NEER data: {input_path}")

    SKIP_ROWS = 2  # Skip the useless stuff at the top

    try:
        df = pd.read_excel(input_path, skiprows=SKIP_ROWS, header=None)

        if df.empty:
            logger.warning("Empty file")
            return

        # Excel structure similar to REER data
        if df.shape[0] < 3:
            logger.error(f"Insufficient header rows")
            return
        
        df.rename(columns={0: "Date"}, inplace=True)
            
        main_titles_row = df.iloc[0].ffill()
        sub_titles_row = df.iloc[1]
        specific_metrics_row = df.iloc[2]

        new_column_headers = []
        for col_idx in range(1, df.shape[1]):
            main_title = str(main_titles_row[col_idx])
            current_sub_title = str(sub_titles_row[col_idx])
            specific_metric = str(specific_metrics_row[col_idx])

            parts = [main_title]
            if pd.notna(sub_titles_row[col_idx]) and current_sub_title.strip() and current_sub_title.lower() != 'nan':
                parts.append(current_sub_title)
            parts.append(specific_metric)
            
            clean_parts = [p for p in parts if p.strip().lower() != 'nan']
            new_column_headers.append(" - ".join(clean_parts))

        if df.shape[0] < 4:
            logger.warning("No data rows found")
            return
            
        data_df = df.iloc[3:].copy()
        
        if data_df.empty:
            logger.warning("Empty data section")
            return

        data_df.columns = ['Date'] + new_column_headers
        data_df = data_df.dropna(subset=['Date'], how='all').reset_index(drop=True)

        if data_df.empty:
            logger.warning("No valid dates")
            return

        # Format Date Column to MM-YYYY format
        logger.info("Converting date format")
        try:
            data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
            data_df['Date'] = data_df['Date'].dt.strftime('%m-%Y')
        except Exception as e:
            logger.warning(f"Date conversion failed: {e}")

        # Keep only the NEER (Dec-1995=100) column and date
        neer_columns = [col for col in data_df.columns if "NEER" in col and "Dec-1995=100" in col]
        
        if not neer_columns:
            # Fallback to any NEER column if the specific one isn't found
            neer_columns = data_df.columns[data_df.columns.str.contains('NEER', case=False, na=False)]
            logger.warning(f"Using fallback NEER columns: {neer_columns}")
        else:
            logger.info(f"Found NEER column: {neer_columns[0]}")
            
        data_df = data_df[['Date'] + list(neer_columns)]

        # Manual rebasing - find the base row (12-2011)
        base_row = data_df[data_df['Date'] == '12-2011']
        
        if base_row.empty:
            logger.warning("Base date 12-2011 not found")
        else:
            # Get base value for the NEER column
            base_value = base_row[neer_columns[0]].iloc[0]
            
            if pd.notna(base_value) and base_value != 0:
                logger.info(f"Rebasing to Dec-2011=100 (base: {base_value:.2f})")
                data_df[neer_columns[0]] = data_df[neer_columns[0]] / base_value * 100
                data_df.rename(columns={neer_columns[0]: "NEER (Dec-2011=100)"}, inplace=True)
            else:
                logger.warning(f"Invalid base value: {base_value}")


        # Remove first 46 rows, first 4 years of data. 
        data_df = data_df.iloc[178:]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data_df.to_excel(output_path, index=False)
        logger.info(f"Saved: {output_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/process_nominal_exchange_rate_data.log"),
            logging.StreamHandler()
        ]
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_file = os.path.join(project_root, 'data', 'preliminary_data', 'georgia_effective_nominal_exchange_rate.xlsx')
    output_file = os.path.join(project_root, 'data', 'processed_data', 'georgia_neer_data_cleaned_processed.xlsx')
    
    logger.info(f"Starting NEER processing")
    process_neer_data(input_file, output_file)
    logger.info("Completed")
