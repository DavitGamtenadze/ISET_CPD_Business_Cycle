import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def rebase_index_columns(df_input, base_date_str, index_column_keyword="Index"):
    """
    Rebases specified index columns in a DataFrame to a new base period.
    The base period's value in the index columns will become 100.

    Args:
        df_input: DataFrame with a 'Date' column
        base_date_str: When we want the new base to be (e.g. "12-1999") 
        index_column_keyword: How to find the index columns (case-insensitive)

    Returns:
        pd.DataFrame: DataFrame with index columns rebased, or original if base date not found.
    """
    logger.info(f"Rebasing indices to {base_date_str}")
    df = df_input.copy() # Work on a copy

    if 'Date' not in df.columns:
        logger.error("Missing Date column")
        return df_input # Return original if 'Date' column is missing

    base_row_df = df[df['Date'].astype(str) == str(base_date_str)]

    if base_row_df.empty:
        logger.warning(f"Base date {base_date_str} not found")
        return df_input
    
    base_row_values = base_row_df.iloc[0]

    rebased_cols_count = 0
    for col_name in df.columns:
        if index_column_keyword.lower() in col_name.lower():
            # Ensure the column is numeric, converting non-numeric to NaN
            original_dtype = df[col_name].dtype
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            if df[col_name].isnull().all():
                logger.warning(f"Column {col_name} all NaN, skipping")
                continue

            base_value = base_row_values.get(col_name)

            if pd.isna(base_value):
                logger.warning(f"Base value NaN for {col_name}")
                continue
            if base_value == 0:
                logger.warning(f"Base value zero for {col_name}")
                continue
            
            logger.info(f"Rebasing {col_name}")
            df[col_name] = (df[col_name] / base_value) * 100
            

            new_col_name = col_name.replace("(Dec-95=100)", "(Dec-11=100)")
            if "GEL/EUR" in col_name:
                new_col_name = col_name.replace("(Dec-2001=100)", "(Dec-11=100)")
            df.rename(columns={col_name: new_col_name}, inplace=True)
            
            rebased_cols_count += 1
            
    if rebased_cols_count > 0:
        logger.info(f"Rebased {rebased_cols_count} columns")
    else:
        logger.info("No columns rebased")
        
    return df

def process_reer_data(input_path, output_path):
    """
    Processes Real Effective Exchange Rate data from Excel.
    The script expects a specific multi-level header structure and transforms it
    into a clean DataFrame with proper column names.
    """
    logger.info(f"Processing REER data: {input_path}")

    SKIP_ROWS = 2  # Skip the useless stuff at the top

    try:
        df = pd.read_excel(input_path, skiprows=SKIP_ROWS, header=None)

        if df.empty:
            logger.warning("Empty file")
            return

        # Excel structure is a bit of a mess:
        # Row 3 -> Main titles 
        # Row 4 -> Sub titles ("Percentage Change over" etc)
        # Row 5 -> Specific metrics ("Index (Dec-95=100)" etc)
        # Row 6+ -> Actual data, finally!

        if df.shape[0] < 3:
            logger.error(f"Insufficient header rows")
            return
            
        main_titles_row = df.iloc[0].ffill()
        sub_titles_row = df.iloc[1] # Don't ffill - NaN might be legit -- EUR/GEL is example
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

        # --- Format Date Column to MM-YYYY format ---
        logger.info("Converting date format")
        try:
            data_df['Date'] = pd.to_datetime(data_df['Date'], errors='coerce')
            data_df['Date'] = data_df['Date'].dt.strftime('%m-%Y')
        except Exception as e:
            logger.warning(f"Date conversion failed: {e}")

        data_df = rebase_index_columns(data_df, base_date_str="12-2011", index_column_keyword="Index")

        # Ditch the columns with "previous" - can be added manually later in an easier manner
        data_df = data_df.loc[:, ~data_df.columns.str.contains('previous')]

        # Remove first 48 rows, first 4 years of data. 
        # data_df = data_df[data_df['Date'] > "12-1999"] <-- This is shit doesn't work and messes up the date index
        data_df = data_df.iloc[180:]

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
            logging.FileHandler("logs/process_real_exchange_rate_data.log"),
            logging.StreamHandler()
        ]
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_file = os.path.join(project_root, 'data', 'preliminary_data', 'georgia_real_effective_exchange_rate.xlsx')
    output_file = os.path.join(project_root, 'data', 'processed_data', 'georgia_reer_data_cleaned_processed.xlsx')
    
    logger.info(f"Starting REER processing")
    process_reer_data(input_file, output_file)
    logger.info("Completed") 