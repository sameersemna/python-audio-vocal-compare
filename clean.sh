#!/bin/bash

# conda activate spleeter

[ -f "$HOME/.bash_export" ] && source "$HOME/.bash_export"

usrDir='/home/sameer'
projDir="$usrDir/Shared/Sync/Private/Work/Projects/video-subtitle-extractor"
workDir="$projDir/output"
backendDir="$projDir/backend"

cores_count=$(nproc --all)
threads_count=$((cores_count - 2))
ffmpeg="/usr/bin/ffmpeg -nostdin -loglevel error -stats -threads $threads_count"
echo "cores_count:$cores_count | threads_count:$threads_count | $ffmpeg"

video_path=$1

video_path_clean=${video_path/.mp4/.cln.mp4}
audio_path=${video_path/.mp4/.mp3}

if [ -f $audio_path ]; then
    echo "Final File exists: $audio_path"
elif [ -f $video_path_clean ]; then
    echo "File exists: $video_path_clean"
else
    # --- Activate Conda Environment ---
    YOUR_ENV="spleeter"
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

    bash $usrDir/Shared/viz/de/clean_vid_sound.sh "$video_path"
    sleep 2
    conda deactivate
fi
