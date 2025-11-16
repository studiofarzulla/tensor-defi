#!/bin/bash
# Run CEX data quality analysis

cd /home/kawaiikali/Documents/Resurrexi/coding-with-buddy/tensor-defi

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run analysis
python3 scripts/analyze_cex_data.py

echo ""
echo "Analysis complete. Check:"
echo "  - data/cex_data_analysis.png"
echo "  - data/cex_data_quality_report.txt"
