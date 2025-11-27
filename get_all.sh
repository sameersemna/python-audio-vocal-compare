#!/bin/bash

# --- Configuration ---
# Specify the path to your file containing the YouTube links
LINKS_FILE="yt_links.csv"

# --- Script Logic ---

# Check if the links file exists
if [ ! -f "$LINKS_FILE" ]; then
  echo "Error: Links file not found at '$LINKS_FILE'"
  exit 1
fi

echo "Starting to process links from $LINKS_FILE..."
echo "---"

# Read the file line by line
# IFS= : Prevents 'read' from trimming leading/trailing whitespace
# -r   : Prevents 'read' from interpreting backslashes as escape sequences
# < "$LINKS_FILE" : Redirects the file's content into the 'while' loop
while IFS= read -r line || [[ -n "$line" ]]; do
  
  # Skip empty lines or lines that are comments (start with #)
  if [ -z "$line" ] || [[ "$line" == \#* ]]; then
    continue
  fi

  echo "Found link: $line"
  cmd="bash get_yt.sh '$line'"  
  echo $cmd
  # eval $cmd
  bash get_yt.sh "$line"
  
  # --- End of Processing Logic ---
  echo "--- Sleeping for 60 seconds before processing the next link ---"
  sleep 60

done < "$LINKS_FILE"

echo "---"
echo "Finished processing all links."