import os
from rembg import remove
from PIL import Image

# Define the input and output directories
input_dir = 'datasets/test/image/'
output_dir = 'datasets/test/image/'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Store the full path of the input and output images
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Open the input image
        input_image = Image.open(input_path).convert("RGBA")  # Use RGBA to keep transparency information

        # Remove the background
        output_image = remove(input_image)

        # Create a white background
        white_bg = Image.new("RGBA", output_image.size, (255, 255, 255, 255))

        # Composite the image with the white background
        output_with_white_bg = Image.alpha_composite(white_bg, output_image)

        # Convert the output image to RGB
        output_with_white_bg = output_with_white_bg.convert("RGB")

        # Save the processed image in the output directory
        output_with_white_bg.save(output_path)

        print(f"Processed and saved: {output_path}")

print("Background removal and replacement completed for all images.")
