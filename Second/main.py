# image_processing_assignment.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # For loading and saving images easily
import os # Added for os.path.exists and checking file extensions

# --- HEIC Handling Setup ---
# Attempt to import Pillow_Heif and register its opener.
# This must be done BEFORE Image.open() is called for HEIC files.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener() # Register the HEIF opener with Pillow
    print("Pillow-Heif successfully registered for HEIC/HEIF support.")
except ImportError:
    print("Warning: Pillow-Heif not found. HEIC/HEIF image support may be limited.")
    print("Please install it: pip install pillow-heif")
except Exception as e:
    print(f"An unexpected error occurred during Pillow-Heif setup: {e}")

import torch # We'll need this for torch.nn.Unfold and other PyTorch operations

def load_and_process_image(image_path="your_image_name.jpg"):
    """
    Loads an image, reshapes it, converts to grayscale, and saves the grayscale version.
    Includes robust handling for .HEIC/.HEIF files using Pillow_Heif.

    Args:
        image_path (str): The path to your local image file.
    """
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: image file not found at {image_path}")
        return None

    img_np = None
    try:
        # Check if the file is a HEIC/HEIF based on extension
        if image_path.lower().endswith(('.heic', '.heif')):
            print("Detected HEIC/HEIF file. Attempting to open with Pillow-Heif.")
            img_pil = Image.open(image_path)
            # Ensure the image is in RGB mode, as HEIC can have other modes
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            img_np = np.array(img_pil)
            print(f"Successfully loaded and converted HEIC/HEIF to RGB NumPy array. Shape: {img_np.shape}")
        else:
            # For other formats (JPG, PNG, etc.), open directly with Pillow
            img_pil = Image.open(image_path)
            img_np = np.array(img_pil)
            print(f"Successfully loaded image with Pillow. Shape: {img_np.shape}")

        print(f"Original image shape (H, W, C): {img_np.shape}")

        # Try making the shape [C, H, W]
        # Assuming the image is RGB (H, W, 3) or RGBA (H, W, 4)
        if img_np.ndim == 3 and (img_np.shape[-1] == 3 or img_np.shape[-1] == 4):
            # If RGBA, convert to RGB first by dropping alpha channel
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3] # Take only RGB channels

            img_chw = img_np.transpose(2, 0, 1) # Changes (H, W, C) to (C, H, W)
            print(f"Reshaped to [C, H, W]: {img_chw.shape}")

            # To demonstrate, reshape back to [H, W, C] from [C, H, W]
            img_hwc_from_chw = img_chw.transpose(1, 2, 0)
            print(f"Reshaped back to [H, W, C] from [C, H, W]: {img_hwc_from_chw.shape}")
        elif img_np.ndim == 2:
            print("Image is 2D (grayscale), skipping [C, H, W] reshape demonstration for color.")
        else:
            print(f"Image has unexpected dimensions ({img_np.ndim}D). Skipping [C, H, W] reshape demonstration.")


        # Convert to grayscale
        # If the image has 3 or 4 channels, use luminance method
        if img_np.ndim == 3 and (img_np.shape[-1] == 3 or img_np.shape[-1] == 4):
            # Ensure it's RGB for dot product if it was RGBA
            if img_np.shape[-1] == 4:
                img_np_rgb = img_np[..., :3]
            else:
                img_np_rgb = img_np
            grayscale_img_np = np.dot(img_np_rgb, [0.2989, 0.5870, 0.1140])
        elif img_np.ndim == 2: # Already grayscale
            grayscale_img_np = img_np
        else:
            print("Warning: Image is not 2D (grayscale) or 3D (color). Grayscale conversion might not be accurate.")
            # Fallback: attempt to take first channel if it's multi-dimensional but not typical RGB/RGBA
            if img_np.ndim > 2 and img_np.shape[-1] >= 1:
                grayscale_img_np = img_np[:, :, 0]
            else:
                # If it's a 1D array or 0-D or other unexpected shape, conversion might fail.
                raise ValueError("Image format not suitable for standard grayscale conversion.")


        print(f"Grayscale image shape (H, W): {grayscale_img_np.shape}")

        # Normalize to 0-255 range and convert to uint8 for saving
        min_val = grayscale_img_np.min()
        max_val = grayscale_img_np.max()
        if max_val - min_val > 0:
            grayscale_img_display = (grayscale_img_np - min_val) / (max_val - min_val) * 255
        else:
            grayscale_img_display = grayscale_img_np # Image is flat (e.g., all black or white)
        grayscale_img_display = grayscale_img_display.astype(np.uint8)


        # Show the result via matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale_img_display, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off') # Turn off axes for clean display
        plt.show()

        # Save a version of this grayscale image as a png, without matplotlib ticks
        grayscale_pil_img = Image.fromarray(grayscale_img_display)
        output_filename = "grayscale_output.png"
        grayscale_pil_img.save(output_filename)
        print(f"Grayscale image saved as {output_filename}")

        return grayscale_img_np # Return the grayscale numpy array for further processing

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None

def add_border_padding(orig: np.ndarray, border_px: int = 0) -> np.ndarray:
    """
    Adds a border of zeros around a 3D numpy array.

    Args:
        orig (np.ndarray): A 3D numpy array of arbitrary dimensions [a, b, c].
                           If input is 2D (H, W), it will be expanded to (H, W, 1).
        border_px (int): A non-negative integer representing the width of the border
                         to add on each side. Defaults to 0 (no border).

    Returns:
        np.ndarray: A new 3D numpy array with dimensions
                    [a + 2 * border_px, b + 2 * border_px, c],
                    containing a copy of the original array values with a border of 0s.
    """
    if not isinstance(orig, np.ndarray):
        raise ValueError("Input 'orig' must be a numpy array.")
    if not isinstance(border_px, int) or border_px < 0:
        raise ValueError("Input 'border_px' must be a non-negative integer.")

    # If the input is 2D (grayscale), expand it to 3D (H, W, 1) to match expected format
    if orig.ndim == 2:
        orig_3d = orig[:, :, np.newaxis]
    elif orig.ndim == 3:
        orig_3d = orig
    else:
        raise ValueError("Input 'orig' must be a 2D or 3D numpy array.")


    original_a, original_b, original_c = orig_3d.shape

    # Calculate new dimensions
    padded_a = original_a + 2 * border_px
    padded_b = original_b + 2 * border_px
    padded_c = original_c # The channel dimension 'c' remains the same

    # Create a new array filled with zeros with the new dimensions
    padded_array = np.zeros((padded_a, padded_b, padded_c), dtype=orig_3d.dtype)

    # Place the original array into the center of the padded array using numpy slicing
    padded_array[border_px : border_px + original_a,
                 border_px : border_px + original_b,
                 :] = orig_3d

    return padded_array

# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure your image path here ---
    # Replace with the actual path to your image file.
    # If using .HEIC, ensure Pillow-Heif is correctly installed (pip install pillow-heif)
    # and working. If you still face issues, convert the .HEIC to .jpg/.png manually.
    my_image_path = "/Users/abiralshakya/Documents/Research/CV_ErichPractice/repvlprojectmatchinginitialchat/IMG_1587.HEIC" # <--- **UPDATE THIS PATH/FILENAME**

    # 1. Load, process, and save grayscale image
    grayscale_image_np = load_and_process_image(my_image_path)

    if grayscale_image_np is not None:
        print("\n--- Testing add_border_padding function with grayscale image ---")

        # Use the add_border_padding function
        # It handles 2D input by expanding it to 3D internally for padding.
        padded_grayscale_1px = add_border_padding(grayscale_image_np, border_px=1)
        print(f"Grayscale image padded with 1px border shape: {padded_grayscale_1px.shape}")

        padded_grayscale_5px = add_border_padding(grayscale_image_np, border_px=5)
        print(f"Grayscale image padded with 5px border shape: {padded_grayscale_5px.shape}")

        # Visualize the padded image
        plt.figure(figsize=(10, 8))
        # .squeeze() removes the channel dimension for display when it's (H, W, 1)
        plt.imshow(padded_grayscale_5px.squeeze(), cmap='gray')
        plt.title('Grayscale Image with 5px Zero Padding')
        plt.axis('off')
        plt.show()

        print("\n--- Proceeding to cross-correlation implementation with torch.nn.Unfold ---")

        # 2. Define the 5x5 filter (kernel)
        filter_5x5 = np.array([
            [1, 0.5, 0, -0.5, -1],
            [2, 1, 0, -1, -2],
            [3, 1.5, 0, -1.5, -3],
            [2, 1, 0, -1, -2],
            [1, 0.5, 0, -0.5, -1]
        ], dtype=np.float32) # Ensure float32 for PyTorch compatibility

        print("\nDefined 5x5 filter:")
        print(filter_5x5)
        print(f"Filter shape: {filter_5x5.shape}")

        # --- Start of torch.nn.Unfold section (as per your assignment prompt) ---
        print("\n--- Demonstrating torch.nn.Unfold ---")

        # Convert grayscale_image_np to a PyTorch tensor, reshape to (N, C, H, W)
        # For a single grayscale image, this is (1, 1, H, W)
        H, W = grayscale_image_np.shape
        grayscale_torch_tensor = torch.Tensor(grayscale_image_np).view(1, 1, H, W)
        print(f"Grayscale image as PyTorch Tensor shape: {grayscale_torch_tensor.shape}")

        # Example from your prompt to understand Unfold's output pattern
        orig_unfold_demo = np.arange(16).reshape(1, 1, 4, 4)
        print(f"\nOriginal Unfold Demo (NumPy): {type(orig_unfold_demo)}")

        orig_unfold_demo_torch = torch.Tensor(orig_unfold_demo)
        print(f"Original Unfold Demo (PyTorch Tensor): {type(orig_unfold_demo_torch)}")

        unfold_op = torch.nn.Unfold(kernel_size=(3, 3), dilation=1, padding=0, stride=1)
        result_unfold_demo = unfold_op(orig_unfold_demo_torch)

        print('\norig_unfold_demo:')
        print(orig_unfold_demo_torch)
        print(orig_unfold_demo_torch.shape)

        print('\nresult_unfold_demo:')
        print(result_unfold_demo.shape)
        # Printing specific columns to show the pattern as in your prompt
        print('result[0, :, 0]:')
        print(result_unfold_demo[0, :, 0])
        print('result[0, :, 1]:')
        print(result_unfold_demo[0, :, 1])
        print('result[0, :, 2]:')
        print(result_unfold_demo[0, :, 2])
        print('result[0, :, 3]:')
        print(result_unfold_demo[0, :, 3])
        # breakpoint() # Uncomment this line if you want to pause execution here for debugging

        print("\n--- Your next task is to use torch.nn.Unfold on your grayscale image,")
        print("    and then perform the cross-correlation with the 5x5 filter. ---")
        print("    Consider the padding needed for the output size to match the input.")