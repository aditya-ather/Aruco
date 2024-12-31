from PIL import Image
import os

# Directory containing the images
image_folder = 'aruco_collage'
output_pdf = 'output.pdf'

# Get all image files in the directory
image_files = [
    '10to30.jpg',
    '35to40.jpg',
    '45to50.jpg',
]

# Sort the files to maintain order
# image_files.sort()

# Open the images and convert them to RGB mode
images = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_files]

# Save the images as a PDF
if images:
    images[0].save(output_pdf, save_all=True, append_images=images[1:])

print(f"PDF saved as {output_pdf}")