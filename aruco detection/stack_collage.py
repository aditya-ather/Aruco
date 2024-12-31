import cv2
import numpy as np

# Define A4 sheet dimensions in pixels (assuming 300 DPI)
A4_WIDTH = 2480
A4_HEIGHT = 3508

# List of image file paths to be stacked
image_paths = [
    # 'collage_10mm.png',
    # 'collage_15mm.png',
    # 'collage_20mm.png',
    # 'collage_25mm.png',
    # 'collage_30mm.png',
    # 'collage_35mm.png',
    # 'collage_40mm.png',
    'collage_45mm.png',
    'collage_50mm.png',
]

# Load images and resize them to the width of A4 sheet
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Error loading image: {path}")
        continue
    height = int(img.shape[0] * (A4_WIDTH / img.shape[1]))
    resized_img = cv2.resize(img, (A4_WIDTH, height))
    images.append(resized_img)

# Calculate the total height of the stacked images
total_height = sum(img.shape[0] for img in images)

# Create a blank A4 sheet
a4_sheet = np.ones((A4_HEIGHT, A4_WIDTH, 3), dtype=np.uint8) * 255

# Stack images on the A4 sheet
current_y = 0
for img in images:
    if current_y + img.shape[0] > A4_HEIGHT:
        print("Warning: Not all images fit on the A4 sheet.")
        break
    a4_sheet[current_y:current_y + img.shape[0], :A4_WIDTH] = img
    current_y += img.shape[0]

# Save the result
cv2.imwrite('stacked_collage.jpg', a4_sheet)
print("Stacked collage saved as 'stacked_collage.jpg'")