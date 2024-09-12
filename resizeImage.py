from PIL import Image
import os

def resize_images_in_folder(folder_path, width=768, height=1024, bit_depth=24):
    # Ensure the target bit depth is 24 (8 bits per channel for RGB)
    if bit_depth != 24:
        raise ValueError("Bit depth must be 24 for RGB images.")

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image by trying to open it
        try:
            with Image.open(file_path) as img:
                # Convert the image to RGB mode (to ensure 24-bit depth)
                img = img.convert("RGB")

                # Resize the image
                img = img.resize((width, height), Image.ANTIALIAS)

                # Save the resized image back to the same file path
                img.save(file_path)

                print(f"Processed and replaced: {filename}")

        except IOError:
            print(f"Skipped non-image file: {filename}")

# Image
Image_path = 'datasets/test/image'
resize_images_in_folder(Image_path, width=768, height=1024, bit_depth=24)
# Cloth
Cloth_path = 'datasets/test/cloth'
resize_images_in_folder(Cloth_path, width=768, height=1024, bit_depth=24)
