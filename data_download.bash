#!/bin/bash

# Data Download Script
# Downloads Georgia economic data from official sources

echo "=============================================="
echo "GEORGIA ECONOMIC DATA DOWNLOAD"
echo "=============================================="

# Set error handling
set -e

# Colors for output (red, green, white only)
RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${WHITE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create preliminary data directory
mkdir -p data/preliminary_data

echo ""
print_step "Downloading quarterly GDP data from GeoStat..."
curl -L -o georgia_quarterly_gdp.xlsx "https://geostat.ge/media/69463/03_GDP-at-Current-Prices.xlsx"
print_success "Quarterly GDP data downloaded"

print_step "Downloading nominal effective exchange rate data..."
curl -L -o georgia_effective_nominal_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/nominal-effective-exchange-rateeng.xlsx?v=y73wd"
print_success "Nominal exchange rate data downloaded"

print_step "Downloading real effective exchange rate data..."
curl -L -o georgia_real_effective_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/real-exchange-rateseng.xlsx?v=maltu"
print_success "Real exchange rate data downloaded"

print_step "Downloading inflation data (2000-2003)..."
curl -L -o georgia_monthly_inflation_2000_2003.xlsx "https://geostat.ge/media/54846/consumer-price-index-to-the-previous-month-2000-2003.xlsx"
print_success "Inflation data (2000-2003) downloaded"

print_step "Downloading inflation data (2004-2025)..."
curl -L -o georgia_monthly_inflation_2004_2025.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/Monthly_inflation_2004_2025.xlsx"
print_success "Inflation data (2004-2025) downloaded"

print_step "Downloading monthly YoY growth rates (2011-2024)..."
curl -L -o georgia_monthly_yoy_growth_rates_2011_2024.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/Rapid_Economic_Growth_Estimates_2011-2025.xlsx"
print_success "YoY growth rates downloaded"

print_step "Downloading monthly GDP real growth rates (2012-2025)..."
curl -L -o georgia_monthly_gdp_real_growth_rates_2012_2025.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/GDP_real_growth.xlsx"
print_success "GDP real growth rates downloaded"

print_step "Moving files to preliminary data directory..."
mv *.xlsx data/preliminary_data/
print_success "All files moved to data/preliminary_data/"

echo ""
echo "=============================================="
echo "DOWNLOAD COMPLETE"
echo "=============================================="

print_success "All economic data downloaded successfully"
echo ""
echo "Downloaded files:"
ls -la data/preliminary_data/

echo ""
echo "=============================================="
