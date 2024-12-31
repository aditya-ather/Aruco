import cv2
from sklearn.decomposition import PCA
import glob
import numpy as np

file = r"tyre\tyre images\IMG_20241220_124835.jpg"
img = cv2.imread(file)

###############
# # K-means
# # Reshape the image to a 2D array of pixels
# pixel_values = img.reshape((-1, 3))
# pixel_values = np.float32(pixel_values)

# # Define criteria and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# k = 2  # Number of clusters
# _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Convert back to 8 bit values
# centers = np.uint8(centers)

# # Map the labels to the center colors
# segmented_image = centers[labels.flatten()]

# # Reshape back to the original image dimension
# img = segmented_image.reshape(img.shape)

# # Display the segmented image
# cv2.imshow('Segmented Image', cv2.resize(segmented_image, (400, 600)))

############
# luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
# segmented_image = cv2.pyrMeanShiftFiltering(luv, sp=21, sr=51)
# img = cv2.cvtColor(segmented_image, cv2.COLOR_LUV2BGR)

###########
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# ###########
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 500)
# if lines is not None:
#     for rho, theta in lines[:, 0]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
############### WatherShed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(binimg, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)

sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8) 
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
# sure foreground 
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that background is not 0, but 1
markers += 1
# mark the region of unknown with zero
markers[unknown == 255] = 0
# watershed Algorithm
markers = cv2.watershed(img, markers)

labels = np.unique(markers)

coins = []
for label in labels[2:]: 

# Create a binary image in which only the area of the label is in the foreground 
#and the rest of the image is in the background 
	target = np.where(markers == label, 255, 0).astype(np.uint8)

# Perform contour extraction on the created binary image
	contours, hierarchy = cv2.findContours(
		target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)
	coins.append(contours[0])

# Draw the outline
img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)

###############


cv2.imshow('img', cv2.resize(img, (400, 600)))
# cv2.imshow('sure_bg', cv2.resize(sure_bg, (400, 600)))
# cv2.imshow('sure_fg', cv2.resize(sure_fg, (400, 600)))
# cv2.imshow('dist', cv2.resize(dist, (400, 600)))
# cv2.imshow('unknown', cv2.resize(unknown, (400, 600)))
# cv2.imshow('Detected Tyre', cv2.resize(img, (400, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()


