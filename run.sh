#!/bin/bash
# Run the Ski Resort Planner with logs redirected to output directory
cd "$(dirname "$0")"
LOG_FILE="output/app_$(date +%Y%m%d_%H%M%S).log"
echo "Starting app, logs: $LOG_FILE"
.venv-skiresort/bin/streamlit run skiresort_planner/app.py --server.port 8502 >> "$LOG_FILE" 2>&1
