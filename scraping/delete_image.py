import os
from PIL import Image

def delete_small_images(folder_path, min_width, min_height):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder not found.")
        return

    # Supported image file type
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip files without valid image extensions
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            print(f"Skipping non-image file: {filename}")
            continue

        # Process the image file
        try:
            with Image.open(file_path) as img:
                width, height = img.size

                # Delete the image if it is smaller than the minimum dimensions
                if width < min_width or height < min_height:
                    os.remove(file_path)
                    print(f"Deleted: {filename} (Size: {width}x{height})")

        except (OSError, ValueError) as e:
            print(f"Could not process {filename}: {e}")

# Example usage
folder_path = "output/Ayam Goreng"  # Replace with the path to your folder
min_width = 100  # Set your minimum width
min_height = 100  # Set your minimum height

delete_small_images(folder_path, min_width, min_height)
