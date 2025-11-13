#!/bin/bash

# bash get_yt.sh 'https://youtu.be/r8zpvs_v6zo?si=aCg4xm_jqufi86nW'

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
export OMP_NUM_THREADS=$threads_count

# Function: activate_conda_env
# Description: Finds the Conda installation base, sources the necessary profile
# script, and attempts to activate the specified Conda environment.
#
# Usage: activate_conda_env <ENVIRONMENT_NAME>
activate_conda_env() {
    # Check if an environment name was provided
    if [ -z "$1" ]; then
        echo "Usage: activate_conda_env <ENVIRONMENT_NAME>" >&2
        return 1
    fi

    local ENV_NAME="$1"
    
    # 1. Find the Conda base installation path using 'conda info --base'
    local CONDA_BASE
    # Suppress error output in case conda isn't immediately available
    CONDA_BASE=$(conda info --base 2>/dev/null)

    if [ -z "$CONDA_BASE" ]; then
        echo "Error: 'conda' base directory could not be determined." >&2
        echo "Please ensure Conda is installed and accessible in your shell's PATH." >&2
        return 1
    fi

    # 2. Source the Conda setup script
    local CONDA_SETUP_SCRIPT="$CONDA_BASE/etc/profile.d/conda.sh"
    if [ -f "$CONDA_SETUP_SCRIPT" ]; then
        # IMPORTANT: Use 'source' or '.' to execute the script in the current shell, 
        # so the 'conda' function is properly loaded.
        . "$CONDA_SETUP_SCRIPT"
    else
        echo "Error: Conda initialization script not found at $CONDA_SETUP_SCRIPT" >&2
        return 1
    fi

    # 3. Activate the specified environment
    echo "Attempting to activate Conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"
    
    # Check for activation success
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to activate Conda environment '$ENV_NAME'. Check the name or existence." >&2
        # We don't return 1 here because the shell hook might still be useful, 
        # but we warn the user.
    else
        echo "Environment '$ENV_NAME' successfully activated."
        echo "Python path: $(which python)"
        python --version
        echo "-----------------------------------"
    fi
}

activate_conda_env youtube
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
    # exit 1
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

# --- 3. Clean Video Audio ---
video_path_clean=${VIDEO_FILE/.mp4/.cln.mp4}
audio_path_clean=${VIDEO_FILE/.mp4/.cln.mp3}
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

# python src/tarteel.py "$video_path_clean"
# python src/tarteel01.py "$video_path_clean"
# python src/tarteel02.py "$video_path_clean"

if [ ! -f "$audio_path_clean" ]; then
    $ffmpeg -i "$video_path_clean" -q:a 0 -map a "$audio_path_clean"
    sleep 2
fi

activate_conda_env captions
sleep 2
# auto-subs transcribe "$video_path_clean" --model small
# modelWhisper='medium'
# modelWhisper='large'
modelWhisper='turbo'
whisper "$audio_path_clean" --model $modelWhisper --threads $threads_count --language ar --task transcribe --output_format all --output_dir "$VIDEO_DIR" --fp16 False
sleep 2
conda deactivate

sleep 2
echo "Subtitle improvement complete."
echo "-----------------------------------"  

# Check if English subtitles were downloaded (required for ffmpeg)
if [ ! -f "$EN_SUB_FILE" ]; then
    echo "Warning: English subtitles (org.en.srt) were not found or did not download."
    echo "Skipping subtitle burn-in process."
    exit 0
fi

echo "Successfully downloaded English subtitles: org.en.srt"

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
    echo "Running ffmpeg to burn subtitles into video..."
    # $ffmpeg -i org.mp4 -vf "subtitles=org.en.srt" -c:a copy -preset fast org.en.mp4
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