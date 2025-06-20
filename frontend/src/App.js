import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Scatter
} from 'recharts';
import './App.css';

// --- Configuration ---
const API_BASE_URL = 'http://localhost:5001/api'; // Assuming Flask runs on port 5001

// --- Helper Components (Placeholders or Simple Implementations) ---

function DataControls({ onFetchData, isLoading, params, setParams }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    onFetchData();
  };

  return (
    <form onSubmit={handleSubmit} className="data-controls">
      <div className="form-group">
        <label htmlFor="filepath">Filepath (optional):</label>
        <input
          type="text"
          id="filepath"
          value={params.filepath}
          onChange={(e) => setParams({ ...params, filepath: e.target.value })}
          placeholder="notebooks/final_gdp_data.xlsx"
        />
      </div>
      <div className="form-group">
        <label htmlFor="sheetName">Sheet Name (optional):</label>
        <input
          type="text"
          id="sheetName"
          value={params.sheetName}
          onChange={(e) => setParams({ ...params, sheetName: e.target.value })}
          placeholder="0 or sheet name"
        />
      </div>
      <div className="form-group">
        <label htmlFor="valueColumn">Value Column (optional):</label>
        <input
          type="text"
          id="valueColumn"
          value={params.valueColumn}
          onChange={(e) => setParams({ ...params, valueColumn: e.target.value })}
          placeholder="e.g., Value_Column_Name"
        />
      </div>
      <div className="form-group">
        <label htmlFor="timestampColumn">Timestamp Column (optional):</label>
        <input
          type="text"
          id="timestampColumn"
          value={params.timestampColumn}
          onChange={(e) => setParams({ ...params, timestampColumn: e.target.value })}
          placeholder="e.g., Date_Column_Name"
        />
      </div>
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Loading Data...' : 'Load Data'}
      </button>
    </form>
  );
}

function MatrixProfileControls({ onCalculateMP, isLoading, windowSize, setWindowSize, hasData }) {
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!hasData) {
      alert("Please load data first before calculating the matrix profile.");
      return;
    }
    onCalculateMP();
  };

  return (
    <form onSubmit={handleSubmit} className="mp-controls">
      <div className="form-group">
        <label htmlFor="windowSize">Window Size (m):</label>
        <input
          type="number"
          id="windowSize"
          value={windowSize}
          onChange={(e) => setWindowSize(parseInt(e.target.value, 10))}
          min="2" // STUMPY typically requires m > 1 or m > 2
          required
        />
      </div>
      <button type="submit" disabled={isLoading || !hasData}>
        {isLoading ? 'Calculating MP...' : 'Calculate Matrix Profile'}
      </button>
    </form>
  );
}

function ChartDisplay({ title, data, xKey, yKeys, mpData, mpIndices, timestamps }) {
  if (!data || data.length === 0) {
    return <p className="chart-placeholder">No data to display for {title}.</p>;
  }

  const chartData = data.map((item, index) => {
    const point = { ...item };
    if (yKeys.includes('value')) point.value = Number(item.value); // Ensure 'value' is numeric
    if (timestamps && timestamps[index]) point.timestamp = timestamps[index]; // Use original timestamps if available

    if (mpData && mpData[index] !== null && !isNaN(mpData[index])) {
      point.matrix_profile = Number(mpData[index]);
    }
    return point;
  });

  // Determine if xKey is time-based or index-based
  const xAxisType = typeof chartData[0]?.[xKey] === 'string' && isNaN(new Date(chartData[0][xKey]).getTime()) ? 'category' : 'number';
  const domain = xAxisType === 'number' ? [dataMin => Math.floor(dataMin), dataMax => Math.ceil(dataMax)] : undefined;


  return (
    <div className="chart-container">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey={xKey}
            type={xAxisType}
            domain={domain}
            tickFormatter={(tick) => {
              if (xAxisType === 'number' && String(tick).length > 7 && !isNaN(new Date(tick).getTime())) { // Heuristic for timestamp
                return new Date(tick).toLocaleDateString();
              }
              return tick;
            }}
          />
          <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
          {yKeys.includes('matrix_profile') && <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />}
          <Tooltip labelFormatter={(label) => {
             if (xAxisType === 'number' && String(label).length > 7 && !isNaN(new Date(label).getTime())) {
                return new Date(label).toLocaleString();
              }
              return label;
          }}/>
          <Legend />
          {yKeys.includes('value') && <Line yAxisId="left" type="monotone" dataKey="value" stroke="#8884d8" dot={false} name="Time Series" />}
          {yKeys.includes('matrix_profile') && mpData && <Line yAxisId="right" type="monotone" dataKey="matrix_profile" stroke="#82ca9d" dot={false} name="Matrix Profile" />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}


// --- Main App Component ---
function App() {
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [matrixProfileData, setMatrixProfileData] = useState(null); // { profile: [], indices: [], timestamps: [] }
  const [windowSize, setWindowSize] = useState(50); // Default window size
  const [dataParams, setDataParams] = useState({
    filepath: '', // Default to empty, backend uses its default
    sheetName: '',
    valueColumn: '',
    timestampColumn: '',
  });
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [isLoadingMP, setIsLoadingMP] = useState(false);
  const [error, setError] = useState(null);

  const handleFetchData = useCallback(async () => {
    setIsLoadingData(true);
    setError(null);
    setMatrixProfileData(null); // Clear previous MP results
    try {
      const payload = {};
      if (dataParams.filepath) payload.filepath = dataParams.filepath;
      if (dataParams.sheetName) payload.sheetName = dataParams.sheetName;
      if (dataParams.valueColumn) payload.valueColumn = dataParams.valueColumn;
      if (dataParams.timestampColumn) payload.timestampColumn = dataParams.timestampColumn;

      const response = await axios.post(`${API_BASE_URL}/data`, payload);
      setTimeSeriesData(response.data || []);
      if (!response.data || response.data.length === 0) {
        setError("No data returned from API or data is empty.");
      }
    } catch (err) {
      console.error("Error fetching time series data:", err);
      setError(err.response?.data?.error || err.message || 'Failed to fetch time series data.');
      setTimeSeriesData([]);
    } finally {
      setIsLoadingData(false);
    }
  }, [dataParams]);

  // Automatically load default data on first render
  useEffect(() => {
    handleFetchData();
  }, [handleFetchData]);


  const handleCalculateMatrixProfile = async () => {
    if (timeSeriesData.length === 0) {
      setError('No time series data available to calculate matrix profile.');
      return;
    }
    setIsLoadingMP(true);
    setError(null);
    try {
      const payload = {
        window_size: windowSize,
        // Option 1: Send data directly
        data: timeSeriesData,
        // Option 2: Send parameters to load data again (if preferred by backend design)
        // filepath: dataParams.filepath,
        // sheet_name: dataParams.sheetName,
        // value_column: dataParams.valueColumn,
        // timestamp_column: dataParams.timestampColumn,
      };
      const response = await axios.post(`${API_BASE_URL}/matrix_profile`, payload);
      setMatrixProfileData(response.data);
    } catch (err) {
      console.error("Error calculating matrix profile:", err);
      setError(err.response?.data?.error || err.message || 'Failed to calculate matrix profile.');
      setMatrixProfileData(null);
    } finally {
      setIsLoadingMP(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Time Series Analysis with Matrix Profile</h1>
      </header>
      <main>
        {error && <div className="error-message">Error: {error}</div>}

        <section className="controls-section">
          <DataControls
            onFetchData={handleFetchData}
            isLoading={isLoadingData}
            params={dataParams}
            setParams={setDataParams}
          />
          <MatrixProfileControls
            onCalculateMP={handleCalculateMatrixProfile}
            isLoading={isLoadingMP}
            windowSize={windowSize}
            setWindowSize={setWindowSize}
            hasData={timeSeriesData.length > 0}
          />
        </section>

        <section className="charts-section">
          <ChartDisplay
            title="Time Series Data"
            data={timeSeriesData}
            xKey="timestamp" // Assuming 'timestamp' is the key for x-axis
            yKeys={['value']} // Key for y-axis
          />
          {matrixProfileData && (
            <ChartDisplay
              title="Matrix Profile"
              // Pass original time series data for x-axis alignment, or use MP timestamps
              data={timeSeriesData}
              xKey="timestamp"
              yKeys={['matrix_profile']} // Key for matrix profile y-axis
              mpData={matrixProfileData.matrix_profile}
              mpIndices={matrixProfileData.profile_indices}
              timestamps={matrixProfileData.timestamps} // Use timestamps from MP response for alignment
            />
          )}
        </section>
      </main>
      <footer className="App-footer">
        <p>Matrix Profile Analysis Tool</p>
      </footer>
    </div>
  );
}

export default App;
