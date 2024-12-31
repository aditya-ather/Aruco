import cv2
import numpy as np

file = r"tyre\tyre images\only_tyre.jpg"
img = cv2.imread(file)

black_threshold = 127
img_copy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)
gray = cv2.GaussianBlur(thresh, (5, 5), 0)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETRIEVE ONLY PARENT/EXTERNAL CONTOURS
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

# hulls = [cv2.convexHull(contours[i]) for i in range(len(contours))]
# cv2.drawContours(img, hulls, -1, (0, 0, 255), 1)

strips = []
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h
	extent = cv2.contourArea(cnt)/(w*h)
	if aspect_ratio > 3 and extent > 0.3:
		strips.append(cnt)
		cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255, 181, 233), 1)

# for cnt in contours:
# 	rect = cv2.minAreaRect(cnt)
# 	box = cv2.boxPoints(rect)
# 	box = np.intp(box)
# 	cv2.drawContours(img,[box],0,(0,255,255),1)

for cnt in strips:
	mask = np.zeros(gray.shape,np.uint8)
	cv2.drawContours(mask,[cnt],0,255,-1) # "-1" means Filled solid white
	pixelpoints = cv2.findNonZero(mask)
	img_copy = cv2.drawContours(img_copy, [pixelpoints], -1, (0, 0, 255), 1)
	color = [0, 0, 0]
	px_count = 0
	for point in pixelpoints:
		x, y = point[0]
		if gray[y, x] > black_threshold:
			color += img[y, x]
			img_copy = cv2.drawContours(img_copy, [point], -1, (255, 0, 0), 1)
			px_count += 1
	mean_color = np.intp(color/px_count)
	print(mean_color)
	# break
	# mean_col = cv2.mean(img,mask = mask)

cv2.imshow('img', cv2.resize(img_copy, (1200, 400)))
cv2.imwrite('tyre_segment.jpg', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


