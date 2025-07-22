import os
import subprocess

# Paths (update as needed)
DEMO_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'demo_ego_blur.py'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/ego_blur_face.jit'))
INPUT_IMAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/tempImagejIGqOK-Photoroom.jpg'))
OUTPUT_IMAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/output_blurred.jpg'))

def blur_faces():
    cmd = [
        'python',
        DEMO_SCRIPT,
        '--face_model_path', MODEL_PATH,
        '--input_image_path', INPUT_IMAGE,
        '--output_image_path', OUTPUT_IMAGE
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    blur_faces()