#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Starting scripts from directory: $SCRIPT_DIR"

# Start the main Raspberry Pi logic script in the background
# Make sure python3 points to the correct Python environment if using virtualenvs
python3 "$SCRIPT_DIR/raspberry_py.py" &
PID1=$! # Get the process ID of the first script

# Start the GUI script in the background
python3 "$SCRIPT_DIR/gui.py" &
PID2=$! # Get the process ID of the second script

echo "raspberry_py.py PID: $PID1"
echo "gui.py PID: $PID2"

# Optional: Function to kill child processes on exit/stop signal
cleanup() {
    echo "Stopping scripts..."
    kill $PID1 $PID2
    echo "Scripts stopped."
}

# Trap signals (like TERM from systemctl stop) and run cleanup
trap cleanup SIGTERM SIGINT

# Wait for both background processes to finish
# This keeps the script running so systemd can monitor it
wait $PID1
wait $PID2

echo "Both scripts have finished."