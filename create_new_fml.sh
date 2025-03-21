#!/bin/bash

# Get all local branches
git fetch
branches=$(git branch --list)

# Extract numbers from branches matching fml<number>
numbers=$(echo "$branches" | grep -oE 'fml[0-9]+' | grep -oE '[0-9]+' || true)

if [ -z "$numbers" ]; then
  echo "No branches matching fml<number> found."
  exit 0
fi

# Find the largest number
max_number=$(echo "$numbers" | sort -nr | head -n 1)

git reset --hard
git checkout -b "fml$((max_number + 1))"
