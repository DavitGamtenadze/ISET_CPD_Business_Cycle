#!/bin/bash

# Data Preparation Pipeline
# Processes Georgia economic data for business cycle analysis

echo "=============================================="
echo "GEORGIA BUSINESS CYCLE DATA PREPARATION"
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

# Create necessary directories
print_step "Creating directory structure..."
mkdir -p data/preliminary_data
mkdir -p data/processed_data  
mkdir -p data/monthly_data
mkdir -p data/quarterly_data
mkdir -p models
mkdir -p logs
print_success "Directories created"

echo ""
echo "=============================================="
echo "PHASE 1: GENERAL DATA PROCESSING"
echo "=============================================="

print_step "Processing nominal exchange rate data..."
if python scripts/process_nominal_exchange_rate_data.py; then
    print_success "Nominal exchange rate processing completed"
else
    print_error "Nominal exchange rate processing failed"
    exit 1
fi

print_step "Processing real exchange rate data..."
if python scripts/process_real_exchange_rate_data.py; then
    print_success "Real exchange rate processing completed"
else
    print_error "Real exchange rate processing failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "PHASE 2: QUARTERLY DATA PROCESSING"
echo "=============================================="

print_step "Processing quarterly GDP data..."
if python scripts/quarterly_data_scripts/process_quarterly_gdp.py; then
    print_success "Quarterly GDP processing completed"
else
    print_error "Quarterly GDP processing failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "PHASE 3: MONTHLY DATA PROCESSING"
echo "=============================================="

print_step "Processing GDP data..."
if python scripts/monthly_data_scripts/process_GDP_data.py; then
    print_success "GDP data processing completed"
else
    print_error "GDP data processing failed"
    exit 1
fi

print_step "Processing inflation data..."
if python scripts/monthly_data_scripts/process_inflation_data.py; then
    print_success "Inflation data processing completed"
else
    print_error "Inflation data processing failed"
    exit 1
fi

print_step "Running Chow-Lin GDP disaggregation..."
if python scripts/monthly_data_scripts/chow_lin_gdp_disaggregation.py; then
    print_success "Chow-Lin disaggregation completed"
else
    print_error "Chow-Lin disaggregation failed"
    exit 1
fi

print_step "Merging final GDP and exchange rate data..."
if python scripts/monthly_data_scripts/merge_gdp_final_data_for_monthly.py; then
    print_success "Final data merge completed"
else
    print_error "Final data merge failed"
    exit 1
fi

print_step "Moving final data to monthly_data folder..."
if [ -f "data/processed_data/final_gdp_exchange_rate_data.xlsx" ]; then
    mv data/processed_data/final_gdp_exchange_rate_data.xlsx data/monthly_data/
    mv data/processed_data/final_gdp_exchange_rate_data.csv data/monthly_data/
    print_success "Final data moved to monthly_data folder"
else
    print_error "Final data file not found"
    exit 1
fi

echo ""
echo "=============================================="
echo "PHASE 4: QUARTERLY ANALYSIS"
echo "=============================================="

print_step "Running quarterly analysis..."
if python scripts/quarterly_data_scripts/quarterly_analysis.py; then
    print_success "Quarterly analysis completed"
else
    print_error "Quarterly analysis failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "DATA PREPARATION COMPLETE"
echo "=============================================="

print_success "All processing steps completed"

echo ""
echo "Output locations:"
echo "- Monthly data: data/monthly_data/"
echo "- Quarterly data: data/quarterly_data/"
echo "- Processed data: data/processed_data/"
echo "- Models: models/"

echo ""
echo "=============================================="
