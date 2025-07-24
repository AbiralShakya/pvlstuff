import cv2
import torch
import torchvision
import numpy as np
import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

LP_MODEL_PATH = "/Users/abiralshakya/Documents/Research/CV_ErichPractice/EgoBlur/ego_blur_lp.jit" 
INPUT_IMAGE_PATH = "/Users/abiralshakya/Documents/Research/CV_ErichPractice/repvlprojectmatchinginitialchat/IMG_3991.HEIC"
INPUT_VIDEO_PATH = "path/to/your/input_video.mp4" 
OUTPUT_IMAGE_PATH = "/Users/abiralshakya/Documents/Research/CV_ErichPractice/EgoBlur/images/output_blurred_lp.jpg"
OUTPUT_VIDEO_PATH = "output_blurred_video.mp4"

BLUR_STRENGTH = 99
DETECTION_THRESHOLD = 0.5 

PROCESS_AS_VIDEO = False

MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH = 829, 1333 

def load_lp_detection_model(model_path):
    """
    Loads a PyTorch TorchScript (.jit) model for license plate detection.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"License plate model not found at: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Successfully loaded license plate detection model from: {model_path} on {device}")
        return model, device
    except RuntimeError as e:
        print(f"Error loading TorchScript model: {e}")
        print("\n--- Troubleshooting Tip (Model Loading) ---")
        print("this often indicates an incompatibility between your PyTorch and Torchvision versions,")
        print("or that the model was saved with a custom Torchvision build.")
        print("1. ensure 'torch' and 'torchvision' are compatible (check PyTorch website).")
        print("2. make sure 'import torchvision' is at the very top of your script.")
        print("---------------------------\n")
        raise 

def read_image_robustly(image_path):
    """
    Reads an image from the given path, handling .HEIC/.HEIF files.
    """
    if not os.path.exists(image_path):
        print(f"Error: image file not found at {image_path}")
        return None

    if image_path.lower().endswith(('.heic', '.heif')):
        try:
            heif_file = Image.open(image_path)
            if heif_file.mode != "RGB":
                heif_file = heif_file.convert("RGB")
            numpy_image = np.array(heif_file)
            image_bgr = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            print(f"Successfully converted and loaded HEIC image: {image_path}")
            return image_bgr
        except Exception as e:
            print(f"Error converting HEIC file {image_path}: {e}")
            return None
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: OpenCV could not load image from {image_path}. it might be corrupted or an unsupported format.")
        else:
            print(f"Successfully loaded image: {image_path}")
        return image

def detect_license_plates(image, model, device, detection_threshold):
    """
    Performs inference using the loaded .jit model to detect license plates.
    """
    original_height, original_width = image.shape[0], image.shape[1]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = torch.from_numpy(image_rgb).float()

    image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = torchvision.transforms.functional.resize(
        image_tensor, 
        size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH), 
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    
    # move tensor to device without adding a batch dimension (model handles internally)
    image_tensor = image_tensor.to(device) 
    
    # debug prints for tensor shape during preprocessing
    print(f"final input tensor shape (to model, should be C, H, W): {image_tensor.shape}")

    with torch.no_grad(): # disable gradient calculation for inference
        try:
            # pass the 3D tensor directly to the model
            model_output = model(image_tensor) 
            
            # debug prints for model output
            print(f"model output type: {type(model_output)}")
            if isinstance(model_output, (list, tuple)):
                for i, out_item in enumerate(model_output):
                    print(f"model output item {i} type: {type(out_item)}, shape: {out_item.shape if hasattr(out_item, 'shape') else 'N/A'}")
            else:
                print(f"model output shape: {model_output.shape if hasattr(model_output, 'shape') else 'N/A'}")
                
        except RuntimeError as e:
            print(f"RuntimeError during model inference: {e}")
            print("\n--- Model Inference Troubleshooting (Input Shape/Preprocessing) ---")
            print("the model's internal layers received an unexpected input.")
            print("1. verify `MODEL_INPUT_HEIGHT`, `MODEL_INPUT_WIDTH`: these must be the exact expected input dimensions for your .jit model.")
            print("2. pixel value range/normalization: ensure pixels are in the correct range (e.g., 0-1 or 0-255) and normalized as expected by the model.")
            print("3. model specifics: consult original code/documentation for precise input signature and preprocessing.")
            print("---------------------------------------\n")
            raise 

    detected_boxes = []
    
    # parse model output to get bounding boxes and scores
    if isinstance(model_output, (list, tuple)) and len(model_output) >= 2:
        pred_boxes = model_output[0].cpu().numpy()
        scores = model_output[1].cpu().numpy()
        
        # scale bounding boxes back to original image dimensions
        scale_x = original_width / MODEL_INPUT_WIDTH
        scale_y = original_height / MODEL_INPUT_HEIGHT

        for i in range(len(scores)):
            if scores[i] > detection_threshold:
                box = pred_boxes[i]
                x_min, y_min, x_max, y_max = box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
                
                detected_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    else:
        print("warning: model output format not recognized. cannot parse detections.")
        print(f"model output type: {type(model_output)}")

    print(f"detected {len(detected_boxes)} license plates.")
    return detected_boxes

def blur_regions(image, bounding_boxes, blur_strength):
    """
    applies a Gaussian blur to specified bounding box regions in an image.
    """
    if blur_strength % 2 == 0: # ensure kernel size is odd
        blur_strength += 1

    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]

        # clamp coordinates to image boundaries
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)

        if x_max > x_min and y_max > y_min:
            roi = image[y_min:y_max, x_min:x_max]
            blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            image[y_min:y_max, x_min:x_max] = blurred_roi
    return image

def process_image():
    """
    processes a single image file for license plate blurring.
    """
    print(f"attempting to process image: {INPUT_IMAGE_PATH}")
    image = read_image_robustly(INPUT_IMAGE_PATH)
    if image is None:
        raise ValueError(f"failed to load or convert image from {INPUT_IMAGE_PATH}. please check the path and file integrity.")

    lp_model, device = load_lp_detection_model(LP_MODEL_PATH)
    
    bounding_boxes = detect_license_plates(image, lp_model, device, DETECTION_THRESHOLD)
    
    blurred_image = blur_regions(image, bounding_boxes, BLUR_STRENGTH)

    # ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_IMAGE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(OUTPUT_IMAGE_PATH, blurred_image)
    print(f"blurred image saved to {OUTPUT_IMAGE_PATH}")

def process_video():
    """
    processes a video file frame by frame for license plate blurring.
    """
    print(f"attempting to process video: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"could not open video from {INPUT_VIDEO_PATH}. please check the path and video format.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for .mp4
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    lp_model, device = load_lp_detection_model(LP_MODEL_PATH)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bounding_boxes = detect_license_plates(frame, lp_model, device, DETECTION_THRESHOLD)
        
        blurred_frame = blur_regions(frame, bounding_boxes, BLUR_STRENGTH)

        out.write(blurred_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"blurred video saved to {OUTPUT_VIDEO_PATH}")

def main():
    if PROCESS_AS_VIDEO:
        process_video()
    else:
        process_image()

if __name__ == "__main__":
    main()