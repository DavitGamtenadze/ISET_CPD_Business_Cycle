import pandas as pd

# Load the policy-rate changes file, using row 1 as header
df = pd.read_excel('/mnt/data/mpirgeo.xlsx', header=1)

# Rename the columns: column 1 → effective_date, column 2 → rate
col1, col2 = df.columns[1], df.columns[2]
df = df.rename(columns={col1: 'effective_date', col2: 'rate'})

# Keep only effective_date and rate
df = df[['effective_date', 'rate']]

# Parse dates (effective_date includes time component)
df['effective_date'] = pd.to_datetime(df['effective_date'])

# Set index and sort
df = df.set_index('effective_date').sort_index()

# Resample to month-end, forward-fill
monthly_rate = df['rate'].resample('M').last().ffill().to_frame('Rate')

# Display full monthly series
