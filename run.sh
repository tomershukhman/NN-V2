#!/bin/bash
clear

# Function to clean up all processes in the current process group.
cleanup() {
    echo "Cleaning up all processes..."
    # Kill every process in the current process group.
    # Using "kill -- -$$" sends the signal to all processes in this group.
    kill -- -$$
    exit 1
}

# Set up trap for SIGINT and SIGTERM.
trap cleanup SIGINT SIGTERM

# Check if CONDA_DEFAULT_ENV is cloudspace
if [ "$CONDA_DEFAULT_ENV" == "cloudspace" ]; then
    echo "Detected cloudspace environment, performing git reset --hard..."
    git reset --hard
    git pull
fi

# Remove the old output directory
rm -rf output

if [ "$CONDA_DEFAULT_ENV" != "cloudspace" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    else
        source .venv/bin/activate
    fi

    # Look for any running tensorboard process and kill it if found.
    PID=$(ps aux | grep "tensorboard --logdir=./outputs/tensorboard" | grep -v grep | awk '{print $2}')
    if [ -n "$PID" ]; then
        echo "Killing TensorBoard process with PID: $PID"
        kill $PID
        echo "TensorBoard process killed."
    else
        echo "No matching TensorBoard process found."
    fi

    # Start TensorBoard in the background.
    nohup tensorboard --logdir=./outputs/tensorboard >/dev/null 2>&1 & disown
fi

# Start your main program.
python main.py

# Wait for background processes before exiting.
wait
