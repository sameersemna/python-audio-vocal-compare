import cv2
import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm # For a progress bar when processing videos

# --- Global variable for the AI pipeline (Conceptual for A Scanner Darkly) ---
# For "A Scanner Darkly" style, a direct pre-trained model like Ghibli Diffusion
# is not commonly available. This section is conceptual for where a specialized
# model would fit.
# If you find a specific model (e.g., a custom fine-tuned Stable Diffusion or GAN),
# you would load it here.
scanner_darkly_pipe = None 
# Example placeholder model ID if one were to exist:
# scanner_darkly_model_id = "your-custom-rotoscope-model/scanner-darkly-diffusion"

def load_scanner_darkly_model():
    """
    Conceptual function to load a specialized AI model for "A Scanner Darkly" style.
    This would ideally be a custom-trained Stable Diffusion or GAN.
    For this example, we'll indicate no specific AI model is loaded,
    and rely on an approximation for demonstration.
    """
    global scanner_darkly_pipe
    try:
        # --- This is where you'd load a real A Scanner Darkly AI model if available ---
        # Example (commented out):
        # import torch
        # from diffusers import StableDiffusionImg2ImgPipeline
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # scanner_darkly_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        #     scanner_darkly_model_id, 
        #     torch_dtype=torch.float16 if device == "cuda" else torch.float32
        # )
        # scanner_darkly_pipe.to(device)
        # print(f"Successfully loaded A Scanner Darkly AI model: {scanner_darkly_model_id}")
        
        print("No specific 'A Scanner Darkly' AI model loaded.")
        print("The stylization will use a computer vision approximation.")
        scanner_darkly_pipe = False # Indicate no AI model is loaded, but allow execution
    except ImportError:
        print("Required AI libraries (torch, diffusers) not installed for a potential AI model.")
        scanner_darkly_pipe = None
    except Exception as e:
        print(f"Error loading conceptual 'A Scanner Darkly' AI model: {e}")
        scanner_darkly_pipe = None


def apply_scanner_darkly_style_to_image(pil_image, edge_strength=100, posterize_colors=4):
    """
    Applies an approximation of the "A Scanner Darkly" rotoscope style to a PIL image.
    This function uses traditional computer vision techniques as a conceptual placeholder
    for a dedicated AI model.

    Args:
        pil_image (PIL.Image.Image): The input image to style.
        edge_strength (int): The sensitivity for edge detection (higher means more edges).
        posterize_colors (int): The number of colors to reduce the image to.

    Returns:
        PIL.Image.Image: The stylized image.
    """
    # Convert PIL Image to OpenCV format (NumPy array) for processing
    cv_image = np.array(pil_image.convert('L')) # Convert to grayscale for Canny

    # 1. Edge Detection (to get the distinct outlines)
    # Canny edge detector: lower threshold, upper threshold
    # The thresholds (e.g., 50, 150) may need tuning based on image content
    edges = cv2.Canny(cv_image, edge_strength // 2, edge_strength)
    
    # Invert edges so lines are black on white (or translucent)
    edges = cv2.bitwise_not(edges) # Invert colors for overlay

    # Convert edges back to PIL image for blending
    edges_pil = Image.fromarray(edges).convert("L")
    
    # 2. Color Reduction (Posterization)
    # Reduces the number of colors in the image, giving it a flatter, painted look
    posterized_image = pil_image.quantize(colors=posterize_colors, method=Image.WEBACCESSIBLE)

    # 3. Blend Posterized Image with Edges
    # Create a new blank image for the final blend
    final_image = Image.new('RGB', posterized_image.size, (255, 255, 255))
    final_image.paste(posterized_image, (0, 0))

    # Overlay the edges (black lines on a white background, or translucent lines)
    # You can adjust opacity of the lines by blending
    # For a stark "A Scanner Darkly" look, a direct paste might be desired
    
    # Convert edges to RGB to blend with color image
    edges_rgb = ImageOps.colorize(edges_pil, black=(0, 0, 0), white=(255, 255, 255))
    
    # Simple alpha composite to overlay lines (adjust alpha for line thickness/darkness)
    # This simulates lines on top of the posterized image.
    final_image = Image.composite(
        Image.new('RGB', posterized_image.size, (0, 0, 0)), # Black for lines
        final_image,
        edges_pil # Use the grayscale edges as alpha mask (0=black, 255=white, so lines are black)
    )

    # Optional: Slightly blur the lines for a more "painted" feel, less sharp digital
    # final_image = final_image.filter(ImageFilter.SMOOTH)

    return final_image


# --- Main Video Processing Function ---
def video_to_scanner_darkly_animation(input_video_path, output_video_path, fps=None, edge_strength=100, posterize_colors=4):
    """
    Converts a video to an "A Scanner Darkly"-like animation by processing frame by frame.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output stylized video.
        fps (int, optional): Frames per second for the output video.
                             If None, uses the original video's FPS.
        edge_strength (int): Sensitivity for Canny edge detection (0-255).
        posterize_colors (int): Number of colors for posterization (2-256).
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_video_path}'. "
              "Check if the path is correct and codecs are installed (e.g., 'ffmpeg').")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = fps if fps is not None else original_fps

    print(f"Processing video: '{input_video_path}'")
    print(f"Original Resolution: {frame_width}x{frame_height}, Original FPS: {original_fps}")
    print(f"Output FPS: {output_fps}")
    print(f"Total Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not create output video writer at '{output_video_path}'. "
              "Check directory permissions or try a different codec ('XVID' might be more compatible).")
        cap.release()
        return

    print("Starting frame processing...")
    for frame_idx in tqdm(range(total_frames), desc="Stylizing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply the approximated "A Scanner Darkly" style
        stylized_pil_image = apply_scanner_darkly_style_to_image(pil_image, edge_strength, posterize_colors)
        
        if stylized_pil_image is None:
            print(f"Warning: Skipping frame {frame_idx + 1} due to style transfer issue.")
            continue

        stylized_frame = cv2.cvtColor(np.array(stylized_pil_image), cv2.COLOR_RGB2BGR)

        if stylized_frame.shape[0] != frame_height or stylized_frame.shape[1] != frame_width:
            stylized_frame = cv2.resize(stylized_frame, (frame_width, frame_height))

        out.write(stylized_frame)

    cap.release()
    out.release()

    print(f"'A Scanner Darkly' style animation successfully saved to '{output_video_path}'")


# --- Example Usage ---
if __name__ == "__main__":
    input_video_file = "input_video.mp4" # <--- IMPORTANT: Change this to your actual video file path
    output_video_file = "scanner_darkly_animated_video.mp4"

    # --- Create a dummy video for testing if input_video.mp4 doesn't exist ---
    if not os.path.exists(input_video_file):
        print(f"'{input_video_file}' not found. Creating a simple dummy video for demonstration.")
        dummy_width, dummy_height = 640, 360
        dummy_fps = 24
        dummy_duration_seconds = 5 # 5 seconds video
        dummy_num_frames = dummy_fps * dummy_duration_seconds
        
        dummy_out = cv2.VideoWriter(input_video_file, cv2.VideoWriter_fourcc(*'mp4v'), dummy_fps, (dummy_width, dummy_height))
        if not dummy_out.isOpened():
             print(f"Error creating dummy video. Please ensure necessary video codecs are installed on your system for MP4V.")
        else:
            for i in range(dummy_num_frames):
                frame = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8) # Black frame
                # Add a simple animation: a moving colored square
                color = (i % 255, (i * 2) % 255, (i * 3) % 255) # Changing color
                center_x = int((i / dummy_num_frames) * (dummy_width - 50)) + 25
                center_y = int((i / dummy_num_frames) * (dummy_height - 50)) + 25
                cv2.rectangle(frame, (center_x, center_y), (center_x + 50, center_y + 50), color, -1) # Draw a colored square
                dummy_out.write(frame)
            dummy_out.release()
            print(f"Dummy video '{input_video_file}' created successfully.")
    # --- End of dummy video creation ---

    # Load the conceptual AI model (will indicate no specific model is used here)
    load_scanner_darkly_model() 

    # Parameters for the "A Scanner Darkly" style approximation
    # Experiment with these values to fine-tune the look
    set_edge_strength = 100 # Higher value -> more detected edges (e.g., 50-200)
    set_posterize_colors = 4 # Fewer colors -> more flat/cartoonish (e.g., 2-8)

    # Run the video conversion
    video_to_scanner_darkly_animation(
        input_video_file, 
        output_video_file, 
        fps=24, 
        edge_strength=set_edge_strength, 
        posterize_colors=set_posterize_colors
    )

    print("\n--- Important Notes for 'A Scanner Darkly' Style ---")
    print("1. **Approximation**: The current `apply_scanner_darkly_style_to_image` function uses")
    print("   computer vision techniques (edge detection, posterization) to *approximate* the style.")
    print("   It is not a true end-to-end AI style transfer trained specifically on 'A Scanner Darkly'.")
    print("   The result may look simplistic compared to the film's complex rotoscoping.")
    print("2. **Dedicated AI Model**: Achieving the exact 'A Scanner Darkerly' look would ideally")
    print("   require a specialized deep learning model, potentially fine-tuned on frames from the movie")
    print("   or similar rotoscoped animation data. Such models are generally not off-the-shelf.")
    print("3. **Temporal Consistency**: Even with image stylization, maintaining smooth temporal")
    print("   consistency in video is a significant challenge. The current approach will process")
    print("   each frame independently, which may lead to 'flickering' or jittery lines/colors.")
    print("   Advanced techniques like optical flow or video-specific style transfer models are needed.")
    print("4. **Parameters**: Experiment with `edge_strength` and `posterize_colors` to modify the visual outcome.")
    print("   - `edge_strength`: Controls the prominence and density of the drawn outlines.")
    print("   - `posterize_colors`: Controls the flatness and number of colors in the fill areas.")
    print("5. **Performance**: Processing video frames on CPU, even with traditional CV, can be slow.")
    print("   Consider reducing input video resolution for faster experimentation.")