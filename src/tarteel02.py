import os
# --- Early CPU control -----------------------------------------------------
# Allow the user to limit CPU usage via an optional --cpus CLI flag or
# environment variable TARTEEL_CPU_THREADS. These must be set before
# importing heavy native libraries (torch, transformers) so that OpenMP/MKL
#/OpenBLAS pick them up during initialization.
import argparse

# Minimal early arg parsing: use parse_known_args so the main parser later
# in the file can still parse its own arguments.
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('--cpus', type=int, default=None,
                           help='Number of CPU threads the process should use')
_early_parser.add_argument('--pin-cores', type=str, default=None,
                           help='Comma-separated CPU cores or ranges to pin to, e.g. "0-3,5"')
_early_args, _remaining_argv = _early_parser.parse_known_args()

# Determine desired CPU thread count (env var overrides to preserve old behaviour)
_env_cpus = os.environ.get('TARTEEL_CPU_THREADS')
if _early_args.cpus is not None:
    NUM_CPU_THREADS = max(1, _early_args.cpus)
elif _env_cpus is not None:
    try:
        NUM_CPU_THREADS = max(1, int(_env_cpus))
    except Exception:
        NUM_CPU_THREADS = 4
else:
    NUM_CPU_THREADS = 4

# Export common BLAS/OMP env vars before heavy imports
os.environ.setdefault('OMP_NUM_THREADS', str(NUM_CPU_THREADS))
os.environ.setdefault('MKL_NUM_THREADS', str(NUM_CPU_THREADS))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(NUM_CPU_THREADS))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(NUM_CPU_THREADS))

# Optionally pin process to specific cores (Linux only)
_pin_cores_spec = _early_args.pin_cores or os.environ.get('TARTEEL_PIN_CORES')
if _pin_cores_spec:
    try:
        def _parse_core_spec(spec: str):
            cores = set()
            for part in spec.split(','):
                part = part.strip()
                if '-' in part:
                    a, b = part.split('-', 1)
                    cores.update(range(int(a), int(b) + 1))
                else:
                    cores.add(int(part))
            return cores

        cores_to_pin = _parse_core_spec(_pin_cores_spec)
        # os.sched_setaffinity exists on Linux; wrap in try/except for portability
        try:
            os.sched_setaffinity(0, cores_to_pin)
        except AttributeError:
            # Not supported on this platform
            pass
    except Exception:
        # If parsing/pinning fails, continue without raising so script still runs
        pass

# Now import heavy modules
import torch
import datetime
# Import the necessary classes from transformers
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, GenerationConfig
import librosa # Use librosa for robust audio loading
import soundfile # A dependency for librosa's mp3/video loading
from transformers import pipeline # Re-importing pipeline for robust segment extraction

# --- Configuration ---
# Use the specific Tarteel model as requested
# MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"
MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
MODEL_ID = "openai/whisper-tiny"
# We still need the base model processor and config to get the correct token IDs
BASE_MODEL_ID = "openai/whisper-tiny"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" 

# --- Utility Functions for Subtitle Formatting ---

def format_timestamp(seconds, format_type='srt'):
    """Converts a duration in seconds to the required subtitle timestamp format."""
    
    # Calculate hours, minutes, seconds, and milliseconds
    td = datetime.timedelta(seconds=seconds)
    total_milliseconds = int(td.total_seconds() * 1000)
    
    # Get hours, minutes, seconds, and milliseconds/microseconds
    milliseconds = total_milliseconds % 1000
    seconds = int(td.total_seconds()) % 60
    minutes = int(td.total_seconds() // 60) % 60
    hours = int(td.total_seconds() // 3600)

    if format_type == 'srt':
        # SRT format: HH:MM:SS,mmm
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    elif format_type == 'vtt':
        # VTT format: HH:MM:SS.mmm (sometimes includes leading zeros for hours even if 0)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    else:
        # Should not happen, but return raw seconds as fallback
        return str(seconds)

def save_srt_file(segments, output_filepath):
    """Generates and saves the transcript in SubRip (.srt) format."""
    
    print(f"Generating SRT file at: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            # *** FIX: Skip segments with missing timestamps (NoneType error prevention) ***
            if segment['start'] is None or segment['end'] is None:
                print(f"Skipping segment {i + 1} due to missing start/end timestamp.")
                continue

            start_time = format_timestamp(segment['start'], 'srt')
            end_time = format_timestamp(segment['end'], 'srt')
            text = segment['text'].strip()

            # SRT entry structure: Index, Timecode, Text
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def save_vtt_file(segments, output_filepath):
    """Generates and saves the transcript in WebVTT (.vtt) format."""
    
    print(f"Generating VTT file at: {output_filepath}")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        # VTT file must start with WEBVTT
        f.write("WEBVTT\n\n")
        for segment in segments:
            # *** FIX: Skip segments with missing timestamps (NoneType error prevention) ***
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

# --- Main Transcription Function (Pipeline + Configuration Fix) ---

def transcribe_file(file_path, output_dir="./output"):
    """
    Transcribes an audio or video file using the robust Hugging Face pipeline,
    manually patching the fine-tuned model's configuration for timestamp support.
    """
    print("--- Initializing Transcription Pipeline ---")
    print(f"Model: {MODEL_ID}")
    print(f"Using device: {DEVICE}")

    try:
        # 1. Load the Processor (Tokenizer + Feature Extractor) from the BASE model
        # The base model's processor knows the correct token IDs for timestamps.
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

        # 2. Load the fine-tuned (Tarteel) model weights
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(DEVICE)
        
        # 3. *** THE CRITICAL FIX: Load GenerationConfig from Base Model ***
        # The Tarteel model is missing the GenerationConfig required for timestamps.
        # We load the full config from the base Whisper model.
        base_generation_config = GenerationConfig.from_pretrained(BASE_MODEL_ID)
        
        # 4. Attach the full GenerationConfig to the Tarteel model instance
        # This tells the model how to generate the special tokens required for timestamps.
        model.generation_config = base_generation_config

        # 5. Initialize the ASR pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model, # Use the model with the fixed generation_config
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=DEVICE,
            chunk_length_s=30, # Important for long audio/video files
        )
        
        # 6. Set up the generation arguments to force Arabic and timestamps
        generate_kwargs = {
            "language": "ar",
            "task": "transcribe",
            "max_new_tokens": 444 # Adjusted token limit
        }

    except Exception as e:
        print(f"Error initializing the pipeline: {e}")
        print("Please ensure all dependencies are correctly installed.")
        return

    print(f"\n--- Starting Transcription of: {os.path.basename(file_path)} ---")
    
    try:
        # 7. Call the pipeline
        print("Starting pipeline transcription (this may take a while)...")
        # We set return_timestamps=True here, which tells the pipeline to handle the segment extraction
        result = pipe(
            file_path, 
            return_timestamps=True, # Requests segment-level timestamps
            generate_kwargs=generate_kwargs
        )

        print("Decoded result:", result)
        
        # 8. Extract segments and full text from the result dictionary
        segments = []
        full_transcript = result.get('text', '')
        
        if 'chunks' in result:
            segments_data = result['chunks']
            for chunk in segments_data:
                # The pipeline result should be clean here
                segments.append({
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text']
                })
        else:
            print("Warning: Pipeline succeeded in generating text but failed to generate segmented chunks.")
            # If chunks are missing, full_transcript still contains the result
            
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        print("Check if the file path is correct and 'ffmpeg' is installed.")
        return

    # --- Output Generation ---
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 1. Plain Text Output
    txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
    save_txt_file(full_transcript, txt_path)

    # 2. SRT Subtitle Output
    if segments:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        save_srt_file(segments, srt_path)
    else:
        print("Warning: Segments not found in the output. Skipping SRT/VTT generation.")

    # 3. VTT Subtitle Output
    if segments:
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
        save_vtt_file(segments, vtt_path)

    print("\n--- Transcription Complete ---")
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
    
    if os.path.exists(INPUT_FILE):
        transcribe_file(INPUT_FILE, output_dir=output_dir)
    else:
        print(f"Error: The input file was not found at the specified path: {INPUT_FILE}")
        print("Please double-check the file path.")