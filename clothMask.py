import os
from transformers import pipeline
import numpy as np
from PIL import Image

# Initialize the pipeline
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device='cpu')

# Define the input and output directories
input_dir = "datasets/test/cloth"
output_dir = "datasets/test/cloth-mask"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only image files
        image_path = os.path.join(input_dir, filename)

        # Apply the segmentation pipeline
        pillow_mask = pipe(image_path, return_mask=True)  # outputs a pillow mask
        pillow_image = pipe(image_path)  # applies mask on input and returns a pillow image


        # Process the mask
        pillow_mask = (np.array(pillow_mask) > 0).astype(np.uint8) * 255
        desired_width, desired_height = 768, 1024
        resized_binary_mask = Image.fromarray(pillow_mask).resize((desired_width, desired_height), Image.NEAREST)

        # Save the processed mask
        output_path = os.path.join(output_dir, filename)  # Save with the same filename in the output directory
        resized_binary_mask.save(output_path, format='JPEG')

print("Processing complete!")
