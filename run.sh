#!/bin/bash

if env | grep -q '^LIGHTNING_TEAMSPACE='; then
    echo "Running in Lightning"
    echo "Git reset"
    git reset --hard HEAD
    echo "Git pull"
    git pull
fi

rm -rf output
python3 trainer.py