import argparse
import pandas as pd

# Map Georgian month names to numbers if needed; extend or replace for English months
MONTH_MAP = {
    "იანვარი": 1, "თებერვალი": 2, "მარტი": 3, "აპრილი": 4,
    "მაისი": 5, "ივნისი": 6, "ივლისი": 7, "აგვისტო": 8,
    "სექტემბერი": 9, "ოქტომბერი": 10, "ნოემბერი": 11, "დეკემბერი": 12
}

def main(infile, outfile, sheet=0):
    # Read raw Excel table
    raw = pd.read_excel(infile, sheet_name=sheet, header=None)
    years = raw.iloc[1, 1:].astype(int).tolist()
    vals = raw.iloc[2:, 1:1+len(years)]
    vals.columns = years
    vals.insert(0, 'month', raw.iloc[2:, 0].tolist())

    # Melt to long form
    long = (vals
            .melt(id_vars='month', var_name='year', value_name='yoy_pct')
            .dropna(subset=['yoy_pct'])
            .assign(month_num=lambda d: d['month'].map(MONTH_MAP)))
    long['date'] = pd.to_datetime(dict(year=long['year'], month=long['month_num'], day=1))
    long['yoy_dec'] = long['yoy_pct'] / 100.0
    long = long.sort_values('date')

    # Rebuild monthly level index
    base_year = long['date'].min().year - 1
    levels = {pd.Timestamp(year=base_year, month=m, day=1): 100.0 for m in range(1,13)}
    for date, g in long.set_index('date')['yoy_dec'].items():
        levels[date] = levels[date - pd.DateOffset(years=1)] * (1.0 + g)
    levels_series = pd.Series(levels).sort_index()
    levels_series = levels_series[levels_series.index >= '2012-01-01']

    # Aggregate to quarters and compute QoQ
    Q = levels_series.resample('Q').sum()
    qoq = Q.pct_change(1).dropna()

    # Output
    df_out = pd.DataFrame({
        'Quarter': Q.index.to_period('Q'),
        'QoQ_growth': qoq.round(4)
    })
    print(f"Saved QoQ growth series to {outfile}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input Excel file')
    parser.add_argument('-o','--output', default='quarterly_qoq.csv', help='Output CSV')
    parser.add_argument('--sheet', default=0, help='Sheet index or name')
    args = parser.parse_args()
    main(args.input, args.output, args.sheet)
