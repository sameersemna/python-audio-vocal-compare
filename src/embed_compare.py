from transformers import AutoProcessor, AutoModel
import torch
import librosa
import numpy as np
from compare_vocals import compare_vocal_features_with_dtw

# Load a pre-trained model and its processor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")

def generate_embeddings(file_path):
    # Load audio and resample to the model's expected sample rate (16kHz)
    audio_input, sample_rate = librosa.load(file_path, sr=16000)

    # The model's processor handles tokenizing and normalization
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)

    # Use the model to get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state

    # The embeddings are what we'll use for comparison
    return embeddings.squeeze().numpy()

# Generate embeddings for your three scenarios
embeddings1 = generate_embeddings("data/001002.mp3")
embeddings2 = generate_embeddings("data/001002_copy.mp3")
embeddings_diff_speaker = generate_embeddings("data/001002_alafasy.mp3")
embeddings_diff_text = generate_embeddings("data/001003.mp3")

# Compare and get your desired scores
same_file_score = compare_vocal_features_with_dtw(embeddings1, embeddings2)
same_text_diff_speaker_score = compare_vocal_features_with_dtw(embeddings1, embeddings_diff_speaker)
diff_text_score = compare_vocal_features_with_dtw(embeddings1, embeddings_diff_text)

print(f"Same File Score: {same_file_score}")
print(f"Same Text, Different Speaker Score: {same_text_diff_speaker_score}")
print(f"Different Text Score: {diff_text_score}")