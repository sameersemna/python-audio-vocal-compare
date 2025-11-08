import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, GenerationConfig
import subprocess
import os
import soundfile as sf
import re

# MODEL_NAME = "tarteel-ai/whisper-tiny-ar-quran"
MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
# MODEL_NAME = "openai/whisper-tiny"
input_file = "/home/sameer/Shared/Sync/Private/Work/Projects/qadrai/shuwayyir.mp4"
basename = os.path.splitext(os.path.basename(input_file))[0]
wav_file = f"{basename}.wav"
output_txt = f"{basename}.txt"
output_srt = f"{basename}.srt"
output_vtt = f"{basename}.vtt"

# Step 1: Convert mp4 to wav, mono, 16kHz (correct ffmpeg order!)
if not os.path.exists(wav_file):
    subprocess.run([
        '/usr/bin/ffmpeg', '-i', input_file,
        '-ac', '1', '-ar', '16000', '-y', wav_file
    ], check=True)

# Step 2: Read audio data from wav (should now be 16kHz!)
audio_data, sample_rate = sf.read(wav_file)
print("Audio data shape:", audio_data.shape, "Sample rate:", sample_rate)
assert sample_rate == 16000, f"Audio sample rate is {sample_rate}, should be 16000!"

# Step 3: Load processor & model and extract features
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME)
audio = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
input_features = audio.input_features

# Step 4: Create robust generation config
gen_config = GenerationConfig()
gen_config.return_timestamps = True
gen_config.no_timestamps_token_id = 50363  # Standard for Whisper
gen_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
# gen_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar")
# gen_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")

# Step 5: Transcribe and debug output
with torch.no_grad():
    output_ids = model.generate(input_features, generation_config=gen_config)
    print("Output IDs:", output_ids)
    result = processor.batch_decode(output_ids, skip_special_tokens=True)
    print("Decoded result:", result)

# Step 6: Parse segments and timestamps from result
segments = []
last_time = None
segment_text = []
for s in result[0].split():
    ts_match = re.match(r"<\|(\d+\.\d+)\|>", s)
    if ts_match:
        if segment_text and last_time is not None:
            segments.append({'timestamp': (last_time, float(ts_match.group(1))), 'text': ' '.join(segment_text)})
            segment_text = []
        last_time = float(ts_match.group(1))
    else:
        segment_text.append(s)
if segment_text and last_time is not None:
    segments.append({'timestamp': (last_time, last_time+2.0), 'text': ' '.join(segment_text)})

print("Segments:", segments)

# Step 7: Write plain text file
with open(output_txt, "w", encoding='utf-8') as f:
    f.write(' '.join([seg['text'] for seg in segments]))
with open(output_txt, "w", encoding='utf-8') as f:
    f.write(result[0])

# Step 8: Subtitle writers
def write_srt(segments, filename):
    def format_time(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(filename, "w", encoding='utf-8') as f:
        for idx, seg in enumerate(segments, 1):
            start, end = seg['timestamp']
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{seg['text'].strip()}\n\n")

def write_vtt(segments, filename):
    def format_time(seconds):
        ms = int((seconds - int(seconds)) * 1000)
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    with open(filename, "w", encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start, end = seg['timestamp']
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{seg['text'].strip()}\n\n")

# Step 9: Export subtitles
write_srt(segments, output_srt)
write_vtt(segments, output_vtt)

print(f"Transcription and subtitles saved: {output_txt}, {output_srt}, {output_vtt}")
