#!/bin/bash
clear
if env | grep -q '^LIGHTNING_TEAMSPACE='; then
    echo "Running in Lightning"
    echo "Git reset"
    git reset --hard HEAD
    echo "Git pull"
    git pull
fi

rm -rf outputs
python3 trainer.py