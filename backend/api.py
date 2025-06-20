from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import stumpy

app = Flask(__name__)

# --- Configuration ---
# Default data filepath, can be overridden by request
DEFAULT_DATA_FILEPATH = "notebooks/final_gdp_data.xlsx"

# --- Data Loading and Preprocessing ---
def load_data(filepath, sheet_name=0, value_column=None, timestamp_column=None):
    """
    Loads and preprocesses data from an Excel file.
    Allows selecting a specific sheet and columns for value and timestamp.
    """
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)

        if value_column is None:
            # If no value column specified, try to use the first numeric column
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in the selected sheet.")
            value_column = numeric_cols[0]
            print(f"No value_column specified, using first numeric column: {value_column}")

        if timestamp_column is not None:
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found.")
            # Attempt to convert to datetime, coercing errors
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
            # Check if conversion was successful for any rows
            if df[timestamp_column].isnull().all():
                raise ValueError(f"Timestamp column '{timestamp_column}' could not be converted to datetime or all values are NaT.")
            # Drop rows where timestamp conversion failed
            df.dropna(subset=[timestamp_column], inplace=True)
            # Sort by timestamp
            df.sort_values(by=timestamp_column, inplace=True)
            # Keep only specified value and timestamp columns
            processed_df = df[[timestamp_column, value_column]].copy()
            processed_df.rename(columns={value_column: 'value', timestamp_column: 'timestamp'}, inplace=True)
        else:
            # If no timestamp column, just use the value column and a default range index
            if value_column not in df.columns:
                raise ValueError(f"Value column '{value_column}' not found.")
            processed_df = df[[value_column]].copy()
            processed_df.rename(columns={value_column: 'value'}, inplace=True)
            processed_df['timestamp'] = range(len(processed_df)) # Default timestamp

        # Ensure 'value' column is numeric
        processed_df['value'] = pd.to_numeric(processed_df['value'], errors='coerce')
        processed_df.dropna(subset=['value'], inplace=True)

        if processed_df.empty:
            raise ValueError("Processed data is empty. Check columns or data content.")

        return processed_df

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")
    except ValueError as ve:
        raise ValueError(f"Error processing data: {ve}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during data loading: {e}")

# --- Matrix Profile Analysis ---
def calculate_matrix_profile(time_series_df, window_size):
    """
    Calculates the matrix profile for the given time series data.
    Expects a DataFrame with a 'value' column.
    """
    if time_series_df is None or time_series_df.empty:
        raise ValueError("Input time series data is empty or None.")
    if 'value' not in time_series_df.columns:
        raise ValueError("Time series DataFrame must contain a 'value' column.")

    time_series = time_series_df['value'].values.astype(float) # Ensure float type

    if len(time_series) < window_size * 2: # Basic check for meaningful MP
        raise ValueError("Time series is too short for the given window size to produce a meaningful matrix profile.")
    if window_size <= 1: # STUMPY requires m > 1
        raise ValueError("Window size must be greater than 1.")

    try:
        mp = stumpy.stump(time_series, m=window_size)
        # mp is an array of arrays: [profile, indices]
        # Return the matrix profile (P) and matrix profile indices (I)
        # Pad with NaNs at the end to match original time series length for easier plotting
        profile = mp[:, 0]
        indices = mp[:, 1]

        nan_padding = np.full(window_size - 1, np.nan)
        full_profile = np.concatenate((profile, nan_padding))
        full_indices = np.concatenate((indices, nan_padding)) # Or handle indices padding as appropriate

        return full_profile.tolist(), full_indices.tolist()
    except Exception as e:
        # Catch specific stumpy errors if known, otherwise general exception
        raise Exception(f"Error calculating matrix profile with stumpy: {e}")

# --- API Endpoints ---
@app.route('/api/data', methods=['POST'])
def get_data():
    """
    API endpoint to get the time series data from the specified Excel file.
    Expects JSON payload with optional 'filepath', 'sheet_name', 'value_column', 'timestamp_column'.
    """
    request_data = request.get_json()
    if not request_data:
        request_data = {} # Use defaults if no JSON body

    filepath = request_data.get('filepath', DEFAULT_DATA_FILEPATH)
    sheet_name = request_data.get('sheet_name', 0) # Default to first sheet
    value_column = request_data.get('value_column')
    timestamp_column = request_data.get('timestamp_column')

    try:
        data = load_data(filepath, sheet_name=sheet_name, value_column=value_column, timestamp_column=timestamp_column)
        return jsonify(data.to_dict(orient='records'))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/api/matrix_profile', methods=['POST'])
def get_matrix_profile():
    """
    API endpoint to calculate and return the matrix profile.
    Expects JSON payload with 'window_size'.
    Optionally, can also accept 'data' (as list of dicts) or
    parameters for `load_data` ('filepath', 'sheet_name', 'value_column', 'timestamp_column').
    """
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Missing JSON payload"}), 400

    window_size = request_data.get('window_size')
    if not isinstance(window_size, int) or window_size <= 0:
        return jsonify({"error": "'window_size' must be a positive integer"}), 400

    try:
        # Option 1: Use data provided directly in the request
        if 'data' in request_data:
            input_data_list = request_data['data']
            if not isinstance(input_data_list, list) or not all(isinstance(item, dict) for item in input_data_list):
                 return jsonify({"error": "'data' must be a list of objects (e.g., [{'timestamp': ..., 'value': ...}])"}), 400
            time_series_df = pd.DataFrame(input_data_list)
            if 'value' not in time_series_df.columns:
                return jsonify({"error": "Provided 'data' must have a 'value' column"}), 400
            if 'timestamp' not in time_series_df.columns: # Add default timestamp if missing
                time_series_df['timestamp'] = range(len(time_series_df))

            # Ensure 'value' is numeric
            time_series_df['value'] = pd.to_numeric(time_series_df['value'], errors='coerce')
            time_series_df.dropna(subset=['value'], inplace=True)
            if time_series_df.empty:
                return jsonify({"error": "Provided 'data' resulted in an empty series after processing."}), 400

        # Option 2: Load data using filepath and other parameters
        else:
            filepath = request_data.get('filepath', DEFAULT_DATA_FILEPATH)
            sheet_name = request_data.get('sheet_name', 0)
            value_column = request_data.get('value_column')
            timestamp_column = request_data.get('timestamp_column')
            time_series_df = load_data(filepath, sheet_name=sheet_name, value_column=value_column, timestamp_column=timestamp_column)

        profile, indices = calculate_matrix_profile(time_series_df, window_size)

        # Include original timestamps in the response for easier frontend plotting
        timestamps = time_series_df['timestamp'].tolist()

        # The matrix profile is shorter than the original series by window_size - 1
        # Align the timestamps with the matrix profile values
        # The first value of the matrix profile corresponds to the window starting at timestamps[0]
        # and ending at timestamps[window_size-1]. So, the "effective" timestamp for a profile
        # point could be considered the start or end of its window.
        # For simplicity, we can return timestamps that align with the start of each window.
        # The last valid timestamp for a matrix profile point is len(timestamps) - window_size.

        # Stumpy's profile is of length N - m + 1.
        # We padded it to N with NaNs.
        # So, the timestamps should match the original series length.

        return jsonify({
            "matrix_profile": profile,
            "profile_indices": indices,
            "timestamps": timestamps # Full original timestamps
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/api/anomalies', methods=['POST'])
def detect_anomalies():
    """
    API endpoint to detect anomalies based on the matrix profile.
    Placeholder: Implement anomaly detection logic (e.g., using top motifs or discords).
    This endpoint will likely need the matrix profile data, or calculate it internally.
    """
    # For now, this remains a placeholder
    # In a real implementation, you'd use the matrix profile (and indices)
    # to find discords (anomalies) or motifs.
    # Example: find the index with the highest matrix profile value for a discord.
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Missing JSON payload"}), 400

    # Assuming matrix_profile and timestamps are passed or calculated
    # matrix_profile_values = request_data.get('matrix_profile')
    # timestamps = request_data.get('timestamps')
    # window_size = request_data.get('window_size') # Needed for context

    # Dummy anomalies for now
    print("Placeholder: Detect anomalies. This part needs full implementation.")
    anomalies_indices = [10, 25] # Example indices in the original time series
    anomaly_points = [
        # {'timestamp': '2023-01-02', 'value': 12, 'anomaly_score': 0.9}, # Replace with actual data
        # {'timestamp': '2023-01-04', 'value': 11, 'anomaly_score': 0.95}
    ]
    # This part would typically involve:
    # 1. Getting or calculating matrix profile.
    # 2. Identifying discords (e.g., highest values in matrix_profile).
    # 3. Mapping these back to original data points.
    return jsonify({
        "message": "Anomaly detection placeholder. Full implementation pending.",
        "potential_discord_indices_in_profile": anomalies_indices, # These would be indices from the matrix profile itself
        "contextual_anomaly_points": anomaly_points
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
