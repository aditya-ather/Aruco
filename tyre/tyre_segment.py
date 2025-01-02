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
rects = []
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	rects.append((x,y,x+w,y+h))
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

centers = []
for cnt in strips:
	M = cv2.moments(cnt)
	cY = int(M["m01"] / M["m00"])
	cX = int(M["m10"] / M["m00"])
	centers.append((cX, cY))
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

for center in centers:
	selected_rects = []
	for rect in rects:
		x1, y1, x2, y2 = rect
		if x1 <= center[0] <= x2:
			selected_rects.append(rect)
		if len(selected_rects) == 3:
			break
	if len(selected_rects) == 3:
		break

for rect in selected_rects:
	x1, y1, x2, y2 = rect
	cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
	


cv2.imshow('img', cv2.resize(img_copy, (1200, 400)))
cv2.imwrite('tyre_segment.jpg', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


