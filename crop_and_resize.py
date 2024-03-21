import os
from PIL import Image

def crop_resize_all_png_in_folders(root_folder, crop_box, new_size):
    # Walk through all folders and subfolders
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                # Construct full file path
                filepath = os.path.join(foldername, filename)
                try:
                    # Open the image
                    img = Image.open(filepath)
                    
                    # Crop the image
                    cropped_img = img.crop(crop_box)
                    
                    # Resize the cropped image
                    resized_img = cropped_img.resize(new_size)
                    
                    # Save the resized image, overwrite the original file
                    resized_img.save(filepath)
                    
                    print(f"Cropped and resized: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

# Define the root folder where you want to crop PNG images
root_folder = r'C:\Users\ACER\Desktop\meaw\dtafafc\test'

# Define the crop box coordinates as (left, upper, right, lower)
# Adjust these values according to your specific cropping requirements
crop_box = (350, 220, 1730, 1600)

# Define the new size for resized images
new_size = (512, 512)

# Call the function to crop and resize all PNG images in all folders
crop_resize_all_png_in_folders(root_folder, crop_box, new_size)
