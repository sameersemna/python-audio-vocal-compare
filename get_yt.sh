#!/bin/bash
set -e # Exit immediately if any command fails

# --- Safety Checks ---
# 1. Check if a URL argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <youtube_url>"
    exit 1
fi

proxyTor='192.168.178.94:9050'
ytDlp='yt-dlp'
# ytDlp="$ytDlp --proxy 'socks5://$proxyTor'"
ytDlp="$ytDlp --cookies-from-browser firefox --sleep-interval 10 --sleep-requests 3"
# ---------------------

[ -f "$HOME/.bash_export" ] && source "$HOME/.bash_export"
usrDir='/home/sameer'
projDir="$usrDir/Shared/Sync/Private/Work/Projects/video-subtitle-extractor"
workDir="$projDir/output"
backendDir="$projDir/backend"

cores_count=$(nproc --all)
threads_count=$((cores_count - 2))
ffmpeg="/usr/bin/ffmpeg -threads $threads_count"
echo "cores_count:$cores_count | threads_count:$threads_count | $ffmpeg"

# --- Activate Conda Environment ---
YOUR_ENV="youtube"
# --- Find and source conda ---
# Use 'conda info --base' to find the base install location
CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ]; then
    echo "Error: conda not found." >&2
    exit 1
fi

# Source the conda setup script
source "$CONDA_BASE/etc/profile.d/conda.sh"
# ---------------------------

echo "Activating '$YOUR_ENV'..."
conda activate "$YOUR_ENV"
# Now you are inside the activated environment
echo "Running commands inside '$YOUR_ENV':"
echo "Python path: $(which python)"
python --version
echo "-----------------------------------"

# 2. Check if required tools are installed
if ! command -v yt-dlp &> /dev/null; then
    echo "Error: yt-dlp is not installed. Please install it to continue."
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install it to continue."
    exit 1
fi

VIDEO_URL=$1

# --- 1. Get Video ID and Set Up Folders ---
echo "Fetching video ID..."
VIDEO_ID=$(yt-dlp --skip-download --get-id "$VIDEO_URL")

if [ -z "$VIDEO_ID" ]; then
    echo "Error: Could not get video ID. Please check the URL."
    exit 1
fi

# Define the main output folder and the specific video folder
OUTPUT_DIR="output"
VIDEO_DIR="$OUTPUT_DIR/$VIDEO_ID"

# Create the directory structure
mkdir -p "$VIDEO_DIR"

echo "Video ID identified: $VIDEO_ID"
echo "Files will be saved in: $VIDEO_DIR"
echo "-----------------------------------"
# Define file paths
VIDEO_FILE="$VIDEO_DIR/org.mp4"
EN_SUB_FILE="$VIDEO_DIR/org.en.srt"
AR_SUB_FILE="$VIDEO_DIR/org.ar.srt"
OUTPUT_VIDEO_FILE="$VIDEO_DIR/org.en.mp4"


# --- 2. Download Video and Subtitles ---
echo "Starting download..."

if [ ! -f "$VIDEO_FILE" ]; then
    # We use yt-dlp's output templating.
    # -o "$VIDEO_DIR/org.%(ext)s" will name:
    #   - Video: org.mp4
    #   - Subs:  org.en.srt, org.ar.srt
    $ytDlp \
        -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
        --write-sub \
        --sub-lang "en,ar" \
        --sub-format "srt" \
        -o "$VIDEO_DIR/org.%(ext)s" \
        "$VIDEO_URL"

    echo "Download complete."
    echo "-----------------------------------"
fi

# Retry downloading English subtitles if not found
if [ ! -f "$EN_SUB_FILE" ]; then
    echo "Retrying download of English subtitles..."
    sleep 2
    $ytDlp --write-auto-sub --sub-lang "en" --sub-format srt -o "$VIDEO_DIR/org.%(ext)s" --skip-download "$VIDEO_URL"
fi
conda deactivate
if [ ! -f "$EN_SUB_FILE" ]; then
    echo "Error: English subtitles (org.en.srt) failed to download after retry."
    exit 1
fi

# Check if the main video file was downloaded
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: Video file ($VIDEO_FILE) failed to download."
    exit 1
fi

# Check if Arabic subtitles were downloaded (optional check)
if [ -f "$AR_SUB_FILE" ]; then
    echo "Successfully downloaded Arabic subtitles: org.ar.srt"
fi

# Check if English subtitles were downloaded (required for ffmpeg)
if [ ! -f "$EN_SUB_FILE" ]; then
    echo "Warning: English subtitles (org.en.srt) were not found or did not download."
    echo "Skipping subtitle burn-in process."
    exit 0
fi

echo "Successfully downloaded English subtitles: org.en.srt"

# --- 3. Clean Video Audio ---
video_path_clean=${VIDEO_FILE/.mp4/.cln.mp4}
if [ ! -f "$video_path_clean" ]; then
    echo "Starting video audio cleaning process..."
    bash clean.sh $VIDEO_FILE
    sleep 2
    echo "Audio cleaning complete."
    echo "-----------------------------------"
fi
if [ -f "$video_path_clean" ]; then
    echo "Successfully created cleaned video file: $video_path_clean"
else
    echo "Error: Cleaned video file ($video_path_clean) was not created."
    exit 1
fi

# --- 3.5. Generate Improved Subtitles with Tarteel ---
echo "Starting subtitle improvement process with Tarteel..."
python src/tarteel.py "$video_path_clean"
sleep 2
echo "Subtitle improvement complete."
echo "-----------------------------------"  

# --- 4. Burn Subtitles with ffmpeg ---
echo "Starting subtitle burn-in process (this may take a while)..."

# We `cd` into the directory to make the ffmpeg command cleaner
# and avoid potential path escaping issues.
cd "$VIDEO_DIR"

# -i org.mp4           -> Input video
# -vf "subtitles=org.en.srt" -> Video filter to burn the SRT file
# -c:a copy            -> Copy the audio stream (no re-encoding)
# -preset fast         -> Prioritize speed over compression size
# org.en.mp4           -> The final output file
if [ ! -f "org.en.mp4" ]; then
    $ffmpeg -i org.mp4 -vf "subtitles=org.en.srt" -c:a copy -preset fast org.en.mp4
fi

if [ $? -eq 0 ]; then
    echo "Successfully created $OUTPUT_VIDEO_FILE with burned-in subtitles."
else
    echo "Error: ffmpeg process failed."
    cd .. # Go back up before exiting
    exit 1
fi

# Go back to the original directory
cd ..

echo "-----------------------------------"
echo "All tasks complete!"