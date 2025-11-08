import os
import torch
import datetime
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, GenerationConfig, pipeline
# import re # Used for advanced text processing (splitting into words)
import math # Used for rounding in alignment logic
import subprocess
import argparse

# --- Configuration ---
# Use the specific Tarteel model for high-quality text
# MODEL_ID_TEXT = "tarteel-ai/whisper-tiny-ar-quran"
MODEL_ID_TEXT = "tarteel-ai/whisper-base-ar-quran"
# Use the stable base model for robust timestamps/segments
# Upgraded from 'tiny' to 'small' for better segmentation quality
# MODEL_ID_SEGMENTS = "openai/whisper-tiny" 
MODEL_ID_SEGMENTS = "openai/whisper-base" 
# MODEL_ID_SEGMENTS = "openai/whisper-small" 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" 

# --- Utility Functions for Subtitle Formatting ---

def format_timestamp(seconds, format_type='srt'):
    """Converts a duration in seconds to the required subtitle timestamp format."""
    
    # Safety check for NoneType error
    if seconds is None:
        return "00:00:00,000" if format_type == 'srt' else "00:00:00.000"
        
    td = datetime.timedelta(seconds=seconds)
    total_milliseconds = int(td.total_seconds() * 1000)
    
    milliseconds = total_milliseconds % 1000
    seconds = int(td.total_seconds()) % 60
    minutes = int(td.total_seconds() // 60) % 60
    hours = int(td.total_seconds() // 3600)

    if format_type == 'srt':
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    elif format_type == 'vtt':
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    else:
        return str(seconds)

def save_srt_file(segments, output_filepath):
    """Generates and saves the transcript in SubRip (.srt) format."""
    
    print(f"Generating SRT file at: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # We start index at 1 for the segment list, ignoring any skipped ones
        valid_segment_count = 1 
        for i, segment in enumerate(segments):
            # Skip segments with missing timestamps
            if segment['start'] is None or segment['end'] is None:
                print(f"Skipping segment {i + 1} due to missing start/end timestamp.")
                continue

            start_time = format_timestamp(segment['start'], 'srt')
            end_time = format_timestamp(segment['end'], 'srt')
            text = segment['text'].strip()

            # SRT entry structure: Index, Timecode, Text
            f.write(f"{valid_segment_count}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
            valid_segment_count += 1

def save_vtt_file(segments, output_filepath):
    """Generates and saves the transcript in WebVTT (.vtt) format."""
    
    print(f"Generating VTT file at: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # VTT file must start with WEBVTT
        f.write("WEBVTT\n\n")
        for segment in segments:
            # Skip segments with missing timestamps
            if segment['start'] is None or segment['end'] is None:
                continue

            start_time = format_timestamp(segment['start'], 'vtt')
            end_time = format_timestamp(segment['end'], 'vtt')
            text = segment['text'].strip()

            # VTT entry structure: Timecode, Text
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def save_txt_file(transcript, output_filepath):
    """Generates and saves the full transcript in plain text (.txt) format."""
    
    print(f"Generating TXT file at: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(transcript)

# --- Alignment Function (The core of the hybrid solution) ---

def align_text_to_segments(full_text_hq, segments_lq):
    """
    Distributes the high-quality full text (HQ) into the time segments (LQ)
    based on the duration of the segments.
    """
    print("\n--- Performing Hybrid Alignment (HQ Text + LQ Timestamps) ---")
    
    # --- CRITICAL FIX: Use simple split for reliable Arabic word tokenization ---
    # The previous regex failed to correctly identify Arabic words, leading to single-character splits.
    # We now split simply on whitespace, which is robust for clean transcript output.
    hq_words = full_text_hq.strip().split()
    total_hq_words = len(hq_words)
    print(f"Total high-quality words (Tarteel): {total_hq_words}")

    # 2. Prepare LQ Segments: Calculate the duration of each segment.
    valid_segments_data = []
    total_duration = 0.0 
    
    for segment in segments_lq:
        # Check for segment validity (start/end must exist and not be None)
        if ('start' not in segment or 'end' not in segment or 
            segment['start'] is None or segment['end'] is None):
            print(f"Warning: Skipping segment due to missing start/end timestamp.")
            continue
            
        segment_duration = segment['end'] - segment['start']
        
        valid_segments_data.append({
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment_duration, # Storing duration instead of word count
            'text': '' # Will be filled with HQ text
        })
        total_duration += segment_duration # Summing total duration
    
    if total_duration == 0.0:
        print("Warning: Base model generated zero total duration. Cannot proceed with alignment.")
        return []

    print(f"Total segment duration (Base Whisper, for weighting): {total_duration:.2f} seconds.")
    
    # Calculate the average words per second (WPS) across the entire transcript
    average_wps = total_hq_words / total_duration
    print(f"Calculated average Words Per Second (WPS): {average_wps:.2f}")

    # 3. Align HQ Words to LQ Segments
    current_hq_word_index = 0
    aligned_segments = []

    for i, segment_data in enumerate(valid_segments_data):
        
        # Calculate the number of words to allocate based on duration
        if i == len(valid_segments_data) - 1:
            # For the last segment, take all remaining words to ensure 100% coverage
            words_to_take = total_hq_words - current_hq_word_index
        else:
            # Allocate words proportionally (duration * WPS), rounding up to prevent loss
            words_to_take_float = segment_data['duration'] * average_wps
            words_to_take = math.ceil(words_to_take_float)
            
        # Ensure we don't exceed the total available words before the loop finishes
        words_to_take = min(words_to_take, total_hq_words - current_hq_word_index)
        
        # Extract the high-quality text slice
        end_index = current_hq_word_index + words_to_take
        hq_segment_words = hq_words[current_hq_word_index:end_index]
        
        # Join the words back into a text string
        hq_segment_text = ' '.join(hq_segment_words)

        aligned_segments.append({
            'start': segment_data['start'],
            'end': segment_data['end'],
            'text': hq_segment_text
        })
        
        current_hq_word_index = end_index

    # Final check is useful in case of floating point errors
    if current_hq_word_index < total_hq_words:
        remaining_words = hq_words[current_hq_word_index:]
        # Add a space only if the last segment text is not empty
        prefix = ' ' if aligned_segments[-1]['text'] else ''
        aligned_segments[-1]['text'] += prefix + ' '.join(remaining_words)
        print(f"Added {len(remaining_words)} remaining words to the final segment due to rounding.")

    print(f"Successfully aligned {len(aligned_segments)} segments.")
    return aligned_segments


# --- Main Transcription Function (The Hybrid Strategy) ---

def transcribe_file(file_path, output_dir="./output"):
    """
    Executes the dual-pipeline transcription: Base model for segments, Tarteel model for text, 
    and then aligns the results.
    """
    print(f"\n--- Starting Hybrid Transcription of: {os.path.basename(file_path)} ---")
    
    # Ensure all models are available before proceeding
    try:
        # Load Processor once from the base model as it has the correct tokenizer/feature extractor
        processor = AutoProcessor.from_pretrained(MODEL_ID_SEGMENTS)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # --- 1. RUN BASE MODEL FOR SEGMENTS (LQ Text, HQ Timestamps) ---
    print("\n[STEP 1/2] Running Base Whisper Model (for Timestamps/Segmentation)...")
    try:
        model_segments = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID_SEGMENTS).to(DEVICE)
        
        pipe_segments = pipeline(
            "automatic-speech-recognition",
            model=model_segments, 
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=DEVICE,
            chunk_length_s=30, 
        )
        
        # We must request language and task here too, as the base model needs it
        generate_kwargs_segments = {
            "language": "ar",
            "task": "transcribe",
            "max_new_tokens": 444
        }
        
        result_segments = pipe_segments(
            file_path, 
            return_timestamps=True, # Critical for getting the segments
            generate_kwargs=generate_kwargs_segments
        )
        print("Decoded result:", result_segments)  # Debugging line to inspect the output structure
        
        # Extract segments (RAW) and full transcript (LQ)
        segments_raw_lq = result_segments.get('chunks', [])
        full_transcript_lq = result_segments.get('text', '')
        
        # *** FIX: Normalize segment structure from pipeline output ***
        segments_lq = []
        for chunk in segments_raw_lq:
            # Pipeline output is {'timestamp': (start, end), 'text': '...'}
            start_time, end_time = chunk.get('timestamp', (None, None))
            segments_lq.append({
                'start': start_time,
                'end': end_time,
                'text': chunk.get('text', '')
            })
        # *** END FIX ***
        
        if not segments_lq:
             print("FATAL ERROR: Base Whisper model failed to generate segments. Cannot proceed with alignment.")
             return
             
        print(f"Base Whisper model successfully generated {len(segments_lq)} segments.")
        
    except Exception as e:
        print(f"An error occurred during Base Model transcription: {e}")
        return

    # --- 2. RUN TARTEEL MODEL FOR FULL TEXT (HQ Text, LQ Timestamps) ---
    print("\n[STEP 2/2] Running Tarteel Model (for High-Quality Full Text)...")
    try:
        # Note: We load the Tarteel model but DO NOT worry about its timestamps,
        # only the full text output.
        model_text = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID_TEXT).to(DEVICE)
        
        # Patch the Tarteel model just in case, though we primarily want the full text
        base_generation_config = GenerationConfig.from_pretrained(MODEL_ID_SEGMENTS)
        model_text.generation_config = base_generation_config
        
        pipe_text = pipeline(
            "automatic-speech-recognition",
            model=model_text,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=DEVICE,
            chunk_length_s=30, 
        )
        
        generate_kwargs_text = {
            "language": "ar",
            "task": "transcribe",
            # We explicitly prevent timestamps here as they clutter the output and are unreliable for this model
            "max_new_tokens": 444
        }
        
        # We set return_timestamps=False here to get the clean, high-quality full text
        result_text = pipe_text(
            file_path, 
            return_timestamps=False, 
            generate_kwargs=generate_kwargs_text
        )
        print("Decoded result:", result_text)  # Debugging line to inspect the output structure
        
        full_transcript_hq = result_text.get('text', '')
        
        if not full_transcript_hq:
             print("FATAL ERROR: Tarteel model failed to generate any text. Cannot proceed.")
             return
             
        print("Tarteel model successfully generated high-quality full transcript.")
        
    except Exception as e:
        print(f"An error occurred during Tarteel Model transcription: {e}")
        return


    # --- 3. ALIGNMENT AND OUTPUT GENERATION ---
    
    # Combine the high-quality text with the high-quality segments
    aligned_segments = align_text_to_segments(full_transcript_hq, segments_lq)
    
    # Final full transcript is the high-quality one
    # Note: We save the *raw* HQ transcript here, the alignment function handles the cleaned version.
    final_full_transcript = full_transcript_hq 

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 1. Plain Text Output (using the HQ Tarteel text)
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_txt_file(final_full_transcript, txt_path)

    # 2. SRT Subtitle Output (using the ALIGNED segments)
    if aligned_segments:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        save_srt_file(aligned_segments, srt_path)
    else:
        print("Warning: Aligned segments are empty. Skipping SRT/VTT generation.")

    # 3. VTT Subtitle Output (using the ALIGNED segments)
    if aligned_segments:
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
        save_vtt_file(aligned_segments, vtt_path)

    print("\n--- Hybrid Transcription Complete ---")
    print(f"Outputs saved to the '{output_dir}' directory.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create the parser object
    parser = argparse.ArgumentParser(
        description="A script to process a single file specified via command line."
    )

    # 2. Add the positional argument (it's required by default)
    parser.add_argument(
        'INPUT_FILE',
        type=str,
        help='The path to the input file to be processed.'
    )

    # 3. Parse the arguments
    args = parser.parse_args()
    
    # The path provided by the user
    # INPUT_FILE = "/home/sameer/Shared/Sync/Private/Work/Projects/qadrai/shuwayyir.cln.mp4"
    INPUT_FILE = args.INPUT_FILE
    output_dir = os.path.dirname(INPUT_FILE)

    wav_file = "/home/sameer/Shared/Sync/Private/Work/Projects/qadrai/process.wav"
    if os.path.exists(wav_file):
        os.remove(wav_file)
    
    if os.path.exists(INPUT_FILE):
        if not os.path.exists(wav_file):
            subprocess.run([
                '/usr/bin/ffmpeg', '-i', INPUT_FILE,
                '-ac', '1', '-ar', '16000', '-y', wav_file
            ], check=True)
        
        transcribe_file(wav_file, output_dir)
    else:
        print(f"Error: The input file was not found at the specified path: {INPUT_FILE}")
        print("Please double-check the file path.")