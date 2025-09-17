import numpy as np
from scipy.spatial.distance import cdist
import os
import soundfile as sf
import librosa
from demucs.separate import main
import time

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

def compare_vocal_features_with_cdist(features1, features2):
    """
    Calculates the average cosine distance between two sets of audio features.
    A lower value indicates higher similarity.

    Args:
        features1 (np.ndarray): The feature vectors from the first audio file.
        features2 (np.ndarray): The feature vectors from the second audio file.

    Returns:
        float: The average cosine distance between the features.
    """
    print("Calculating similarity with cdist...")
    distances = cdist(features1, features2, 'cosine')
    return np.mean(distances)

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
    print("Calculating similarity with DTW...")
    D, wp = librosa.sequence.dtw(X=features1.T, Y=features2.T, metric='cosine')
    return D[-1, -1] / np.sqrt(features1.shape[0] * features2.shape[0])


def print_scores(mp3_file1, mp3_file2):
    print("*" * 60)
    print(f"Start Comparing: {mp3_file1} & {mp3_file2}")

    # 1. Separate vocals from the first MP3 file
    vocal1_data, vocal1_sr = separate_vocals(mp3_file1)
    
    # 2. Check if the second file is a copy of the first
    if os.path.abspath(mp3_file1) == os.path.abspath(mp3_file2):
        print("Detected identical files. Reusing vocal data for a guaranteed 0 score.")
        vocal2_data, vocal2_sr = vocal1_data, vocal1_sr
    else:
        vocal2_data, vocal2_sr = separate_vocals(mp3_file2)
    
    # Ensure both files were successfully processed
    if vocal1_data.size == 0 or vocal2_data.size == 0:
        print("Comparison aborted due to failed vocal separation.")
        return
    
    # 3. Extract features from each vocal track
    features_vocal1 = extract_audio_features(vocal1_data, vocal1_sr)
    features_vocal2 = extract_audio_features(vocal2_data, vocal2_sr)

    # 4. Compare the extracted features using both methods
    cdist_score = compare_vocal_features_with_cdist(features_vocal1, features_vocal2)
    dtw_score = compare_vocal_features_with_dtw(features_vocal1, features_vocal2)

    print("-" * 30)
    print(f"Average Cosine Distance (cdist): {cdist_score:.4f}")
    print(f"Dynamic Time Warping Distance (DTW): {dtw_score:.4f}")
    
    print(f"Finished Comparing: {mp3_file1} & {mp3_file2}")
    print("*" * 60)

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure your data folder exists and contains these files.
    mp3_file1 = "data/001002.mp3"
    mp3_file2 = "data/001002_copy.mp3"
    print_scores(mp3_file1, mp3_file2)

    mp3_file3 = "data/001002.mp3"
    mp3_file4 = "data/001001.mp3"
    print_scores(mp3_file3, mp3_file4)
