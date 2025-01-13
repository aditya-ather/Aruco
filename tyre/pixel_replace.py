import cv2
import numpy as np

# Load the image
image_path = r'tyre\pattern1\frame0.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the range of colors to be replaced
lower_bound = np.array([176, 125, 165])
upper_bound = np.array([216, 165, 205])

# Create a mask for the pixels in the specified range
mask = cv2.inRange(image, lower_bound, upper_bound)
cv2.imshow('vd', mask)
# Replace the pixels in the range with the new color
image[mask != 0] = [255, 255, 255]

# Save the modified image
cv2.imwrite('path_to_save_modified_image.jpg', image)

# Display the original and modified images (optional)
# cv2.imshow('Original Image', cv2.imread(image_path))
# cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()