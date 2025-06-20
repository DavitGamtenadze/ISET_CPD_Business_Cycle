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
curl -L -o georgia_quarterly_gdp.xlsx "https://geostat.ge/media/71443/01_Output.xlsx"
print_success "Quarterly GDP data downloaded"

print_step "Downloading nominal effective exchange rate data..."
curl -L -o georgia_effective_nominal_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/nominal-effective-exchange-rateeng.xlsx?v=y73wd"
print_success "Nominal exchange rate data downloaded"

print_step "Downloading real effective exchange rate data..."
curl -L -o georgia_real_effective_exchange_rate.xlsx "https://nbg.gov.ge/fm/%E1%83%A1%E1%83%A2%E1%83%90%E1%83%A2%E1%83%98%E1%83%A1%E1%83%A2%E1%83%98%E1%83%99%E1%83%90/exchange_rates/eng/real-exchange-rateseng.xlsx?v=maltu"
print_success "Real exchange rate data downloaded"

print_step "Downloading inflation data (2004-2025)..."
curl -L -o georgia_monthly_inflation_2004_2025.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/Monthly_inflation_2004_2025.xlsx"
print_success "Inflation data (2004-2025) downloaded"

print_step "Downloading monthly GDP real growth rates (2012-2025)..."
curl -L -o georgia_monthly_gdp_real_growth_rates_2012_2025.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/GDP_real_growth.xlsx"
print_success "GDP real growth rates downloaded"

print_step "Downloading quarterly weighted deposit interest rates..."
curl -L -o georgia_quarterly_weighted_deposit_interest_rates.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/annual-weighted-deposit-interest-rates-dwirgeo.xlsx"
print_success "Weighted deposit interest rates downloaded"

print_step "Downloading Business Confidence Index..."
curl -L -o georgia_business_confidence_index.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/BCI.xlsx"
print_success "Business Confidence Index downloaded"

print_step "Downloading monetary policy rate changes..."
curl -L -o georgia_monthly_monetary_policy_rate_changes.xlsx "https://cpd-iset-ba-thesis.s3.eu-north-1.amazonaws.com/monetary_policy_rates.xlsx"
print_success "Monetary policy rate changes downloaded"

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
