#!/bin/zsh
setopt extended_glob  # Enable advanced globbing
cd data/train2017

while true; do
  clear
  echo "File count: $(print -l -- *(D.) | wc -l)"
  sleep 10
done
