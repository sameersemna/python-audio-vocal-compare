import numpy as np
from scipy.spatial.distance import cdist
import os
import soundfile as sf
import librosa
from demucs.separate import main
import time
import hashlib

from common import separate_vocals

def extract_audio_features(audio_data, sample_rate):
    """
    Extracts audio features like MFCCs.

    Args:
        audio_data (np.ndarray): Raw audio data.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        np.ndarray: A 2D array of audio features.
    """
    # Use a small number of features for this example
    n_mfcc_count = 13 
    
    if len(audio_data) < 2048:
        print(f"Skipping: signal is too short ({len(audio_data)} samples)")
        return np.array([[]])
        
    print(f"Extracting features from audio data... Sample rate: {sample_rate}")
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc_count)
    return mfccs.T

def compare_vocal_features_with_dtw(features1, features2):
    """
    Calculates the similarity between two sets of audio features using Dynamic Time Warping (DTW).
    A lower value indicates higher similarity.

    Args:
        features1 (np.ndarray): The feature vectors from the first audio file.
        features2 (np.ndarray): The feature vectors from the second audio file.

    Returns:
        float: The DTW distance between the feature matrices.
    """
    print("Calculating acoustic similarity with DTW...")
    D, wp = librosa.sequence.dtw(X=features1.T, Y=features2.T, metric='cosine')
    return D[-1, -1] / np.sqrt(features1.shape[0] * features2.shape[0])


def print_scores(mp3_file1, mp3_file2):
    print("*" * 60)
    print(f"Start Comparing: {mp3_file1} & {mp3_file2}")
    
    # Generate a hash for content-based identity check
    file1_hash = hashlib.md5(open(mp3_file1, 'rb').read()).hexdigest()
    file2_hash = hashlib.md5(open(mp3_file2, 'rb').read()).hexdigest()

    # 1. Separate vocals from the first MP3 file
    vocal1_data, vocal1_sr = separate_vocals(mp3_file1)
    
    # 2. Check if the second file is a bit-for-bit copy of the first
    if file1_hash == file2_hash:
        print("Detected identical files via hash. Reusing vocal data for a guaranteed 0 acoustic score.")
        vocal2_data, vocal2_sr = vocal1_data, vocal1_sr
    else:
        vocal2_data, vocal2_sr = separate_vocals(mp3_file2)
    
    # Ensure both files were successfully processed
    if vocal1_data.size == 0 or vocal2_data.size == 0:
        print("Comparison aborted due to failed vocal separation.")
        return
    
    # 3. Extract features
    features_vocal1 = extract_audio_features(vocal1_data, vocal1_sr)
    features_vocal2 = extract_audio_features(vocal2_data, vocal2_sr)

    # 4. Compare the extracted features with DTW
    dtw_score = compare_vocal_features_with_dtw(features_vocal1, features_vocal2)
    
    print("-" * 30)
    print(f"Acoustic Similarity Score (DTW): {dtw_score:.4f}")
    
    print(f"Finished Comparing: {mp3_file1} & {mp3_file2}")
    print("*" * 60)

# --- Example Usage ---
if __name__ == "__main__":
    # Scenario 1: Identical Speaker, Same Text (and same file)
    mp3_file1 = "data/001002.mp3"
    mp3_file2 = "data/001002_copy.mp3"
    print_scores(mp3_file1, mp3_file2)

    # Scenario 2: Different Speaker, Same Text
    # We will assume 001001.mp3 is Speaker 1 and 001002.mp3 is Speaker 2.
    # Both are reading the same text.
    mp3_file3 = "data/001001.mp3"
    mp3_file4 = "data/001002.mp3"
    print_scores(mp3_file3, mp3_file4)

    # Scenario 3: Different Speaker, Different Text
    # We will simulate this by comparing two different files where the DTW score should be higher.
    mp3_file5 = "data/001003.mp3"
    mp3_file6 = "data/001002.mp3"
    print_scores(mp3_file5, mp3_file6)

    mp3_file5 = "data/001002_abdussamad.mp3"
    mp3_file6 = "data/001002.mp3"
    print_scores(mp3_file5, mp3_file6)

    mp3_file5 = "data/001002_alafasy.mp3"
    mp3_file6 = "data/001002.mp3"
    print_scores(mp3_file5, mp3_file6)
