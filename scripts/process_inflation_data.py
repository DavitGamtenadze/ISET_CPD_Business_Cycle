import pandas as pd
import os

def roman_to_int(roman):
    '''Converts Roman numeral (I-XII) to a two-digit month string (01-12).'''
    roman_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
        'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12
    }
    # Handle potential float inputs from Excel (e.g., XII.0)
    val_str = str(roman).strip().upper()
    if '.' in val_str:
        val_str = val_str.split('.')[0]
        
    val = roman_map.get(val_str)
    if val:
        return f"{val:02d}"
    print(f"Warning: Invalid Roman numeral '{roman}'. Assigning 'INVALID_MONTH'.")
    return "INVALID_MONTH"

def process_inflation_data(input_path, output_path):
    '''Processes the Georgia monthly inflation Excel data.
    The input file has complex structure, which is is mostly used by excel specialists.
    This function processes the data into a more usable format for ML.
    '''
    # Configuration constants for Excel structure
    SKIP_ROWS = 3
    YEAR_COLUMN_INDICES = [1, 13, 25, 37]  # 0-indexed in df after skiprows
    YEAR_VALUES_ROW_IDX = 0                # 0-indexed row in df for year values
    MONTH_HEADERS_ROW_IDX = 1              # 0-indexed row in df for Roman month headers
    DATA_START_ROW_IDX = 2                 # 0-indexed row in df where category data begins
    CATEGORY_COLUMN_IDX = 0                # 0-indexed column in df for category labels

    try:
        df = pd.read_excel(input_path, sheet_name=0, skiprows=SKIP_ROWS, header=None)
        
        # Clear the specified cell (original A4, now df.iloc[0,0])
        df.iloc[YEAR_VALUES_ROW_IDX, CATEGORY_COLUMN_IDX] = None 
        
        processed_data_frames = []
        month_roman_headers_from_df = df.iloc[MONTH_HEADERS_ROW_IDX, :].values
        # Extract all potential category labels once
        category_labels = df.iloc[DATA_START_ROW_IDX:, CATEGORY_COLUMN_IDX].reset_index(drop=True)

        for year_col_idx in YEAR_COLUMN_INDICES:
            if year_col_idx >= df.shape[1]:
                print(f"Warning: Year column index {year_col_idx} is out of bounds for the DataFrame. Skipping this block.")
                continue

            year_value = df.iloc[YEAR_VALUES_ROW_IDX, year_col_idx]
            if pd.isna(year_value):
                print(f"Warning: Year value at column {year_col_idx} (row {YEAR_VALUES_ROW_IDX}) is NaN. Skipping this block.")
                continue
            year_value = int(year_value)

            # Define column range for the current year's month data
            month_data_cols_start = year_col_idx
            month_data_cols_end = year_col_idx + 12 # Standard 12 months
            if month_data_cols_end > df.shape[1]:
                # Adjust if the DataFrame ends before 12 months for this year
                month_data_cols_end = df.shape[1] 
            
            current_block_month_data_indices = range(month_data_cols_start, month_data_cols_end)
            if not list(current_block_month_data_indices):
                 print(f"Warning: No month columns to process for year {year_value}. Skipping this block.")
                 continue

            # Extract data values for the current year block
            block_df_values = df.iloc[DATA_START_ROW_IDX:, list(current_block_month_data_indices)].reset_index(drop=True)
            
            # Get Roman month headers corresponding to these data columns
            block_month_roman_numerals = [month_roman_headers_from_df[mc_idx] for mc_idx in current_block_month_data_indices]
            
            numerical_month_column_names = [roman_to_int(rn) for rn in block_month_roman_numerals]
            block_df_values.columns = numerical_month_column_names
            
            # Insert category labels (they should align row-wise with block_df_values)
            block_df_values.insert(0, 'Category', category_labels)

            # Melt the DataFrame from wide to long format
            melted_block = block_df_values.melt(id_vars=['Category'], var_name='Month_Num_Str', value_name='Value')
            
            # Filter out rows where Roman numeral conversion failed (marked as 'INVALID_MONTH')
            melted_block = melted_block[melted_block['Month_Num_Str'] != "INVALID_MONTH"]
            if melted_block.empty:
                print(f"Warning: No valid months after filtering for year {year_value}. Skipping this block.")
                continue
                
            melted_block['Year-Month'] = f"{year_value}-" + melted_block['Month_Num_Str']
            
            final_block = melted_block[['Year-Month', 'Category', 'Value']]
            processed_data_frames.append(final_block)

        if not processed_data_frames:
            print("Warning: No data was processed. Output file will be empty or not generated.")
            return

        final_df = pd.concat(processed_data_frames, ignore_index=True)
        
        # Data cleaning before pivoting
        final_df = final_df.dropna(subset=['Category']) # Remove rows where Category is NaN
        final_df['Category'] = final_df['Category'].astype(str) # Ensure Category is string type
        final_df = final_df.drop_duplicates(subset=['Year-Month', 'Category'], keep='first')
        
        if final_df.empty:
            print("Warning: DataFrame is empty after cleaning steps. No output file will be generated.")
            return
            
        # Pivot the table to get Year-Month as index and Categories as columns
        transposed_df = final_df.pivot(index='Year-Month', columns='Category', values='Value').reset_index()
        transposed_df = transposed_df.sort_values(by='Year-Month').reset_index(drop=True)

        # Add _CPI suffix to category column names (all columns except the first 'Year-Month' column)
        if not transposed_df.empty:
            new_column_headers = [transposed_df.columns[0]] # Keep 'Year-Month' as is
            for category_column_name in transposed_df.columns[1:]:
                new_column_headers.append(f"{str(category_column_name)}_CPI")
            transposed_df.columns = new_column_headers
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        transposed_df.to_excel(output_path, index=False)
        print(f"Successfully processed data and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc() # Keep for detailed error reporting

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'data', 'preliminary_data', 'georgia_monthly_inflation_2000_2003.xlsx')
    output_file = os.path.join(project_root, 'data', 'processed_data', 'georgia_monthly_inflation_2000_2003_processed.xlsx')
    
    process_inflation_data(input_file, output_file) 