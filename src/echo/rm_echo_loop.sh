#!/bin/bash

dirUser="/home/sameer"
dirProj="$dirUser/Shared/Sync/Private/Work/Projects/qadrai"
echo "Starting loop..."

# Method 1: Brace Expansion (Recommended)
# The '01' tells bash to pad numbers with a zero if they are single digits.
for i in {01..02}; do
    echo "Processing: $i"
    
    fileNom=$i

	bash rm_echo.sh "$fileNom.mp3"
done

echo "Loop finished."
echo "-----------------------------------"