import numpy as np
import os
import librosa
from demucs.separate import main
import time

def get_filename_without_extension(file_path):
    """
    Returns the filename from a given path, without its extension.
    
    Args:
        file_path (str): The full path to the file.
        
    Returns:
        str: The filename without the extension.
    """
    # First, get the filename from the full path
    base_name = os.path.basename(file_path)
    
    # Then, split the filename and its extension
    filename, extension = os.path.splitext(base_name)
    
    return filename

def separate_vocals(mp3_file_path):
    """
    Separates the vocal track from an MP3 file using a source separation
    library like Demucs.

    Args:
        mp3_file_path (str): The path to the input MP3 file.

    Returns:
        tuple: A tuple containing the raw vocal audio data and sample rate.
    """
    # print(f"Attempting to separate vocals from: {mp3_file_path}")
    mp3_file_name = get_filename_without_extension(mp3_file_path)
    
    # Demucs output path is typically `separated/htdemucs/{track_name}/vocals.mp3`
    demucs_output_path = f"separated/htdemucs/{mp3_file_name}"
    vocals_file_path = os.path.join(demucs_output_path, "vocals.mp3")

    # Check if the separated vocals file already exists
    if not os.path.exists(vocals_file_path):
        print(f"'{vocals_file_path}' does not exist. Running Demucs...")
        try:
            # Run Demucs with a wait to ensure the file is created
            main(['--mp3', '--two-stems=vocals', mp3_file_path])
            # Give Demucs a moment to finish writing the file
            time.sleep(2) 
        except Exception as e:
            print(f"An error occurred during vocal separation: {e}")
            return np.array([]), 0

    if os.path.exists(vocals_file_path):
        # print(f"'{vocals_file_path}' successfully separated. Loading...")
        try:
            # Load the separated vocal audio
            vocal_audio, sample_rate = librosa.load(vocals_file_path, sr=None)
            return vocal_audio, sample_rate
        except Exception as e:
            print(f"An error occurred loading the separated vocal file: {e}")
            return np.array([]), 0
    else:
        print(f"Failed to find separated vocals file at: {vocals_file_path}")
        return np.array([]), 0