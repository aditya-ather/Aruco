import cv2
import numpy as np

# Load the image
img = cv2.imread(r'tyre\tyre images\IMG_20241220_124835.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color ranges (adjust these as needed)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Create masks for each color
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Find contours in each mask
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours_yellow))
# Process contours (e.g., filter, draw bounding boxes)
for cnt in contours_red:
    # Calculate average HSV color of the contour
    avg_hsv = cv2.mean(hsv, mask=mask_red)
    # Identify color based on average HSV
    if avg_hsv[0] < 10:
        # print("Red line detected")
        pass

# Draw bounding boxes around red contours
for cnt in contours_yellow:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 255, 255), 1)

# Draw bounding boxes around blue contours
# for cnt in contours_blue:
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Detected Colors', cv2.resize(img, (1000, 400)))
cv2.imwrite(r'tyre\tyre images\124835.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()