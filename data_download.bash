# Quarterly GDP data from AskGov, GeoStat only had data from 2010
curl -L -o georgia_quarterly_gdp.xlsx "https://askgov.ge/ka/request/2971/response/4047/attach/8/.xlsx?cookie_passthrough=1"

# Nominal effective exchange rate, monthly data
curl -L -o georgia_effective_nominal_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/nominal-effective-exchange-rateeng.xlsx?v=y73wd"

# Real effective exchange rate, monthly data
curl -L -o georgia_real_effective_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/real-exchange-rateseng.xlsx?v=maltu"

# Monthly inflation data from 2000 to 2003
curl -L -o georgia_monthly_inflation_2000_2003.xlsx "https://geostat.ge/media/54846/consumer-price-index-to-the-previous-month-2000-2003.xlsx"

# Monthly inflation data from 2004 to 2025 / already in percentages needs to be divided by 100
curl -L -o georgia_monthly_inflation_2004_2025.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/Monthly_inflation_2004_2025.xlsx"


# Move all files ending with .xlsx to data/preliminary_data
mv *.xlsx data/preliminary_data/
