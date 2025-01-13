import cv2
import numpy as np

def subpixel(image, corners):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, 'DICT_4X4_50'))
aruco_params = cv2.aruco.DetectorParameters()
# aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# image = cv2.imread(r"C:\Users\aditya.raj\Downloads\m1.jpg")
image = cv2.imread(r"C:\Users\aditya.raj\Downloads\output.png")
# image = cv2.resize(image, (1200, 600))
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
(corners1, ids, rejected) = detector.detectMarkers(image)
print(len(corners1))
if len(corners1) > 0:
	ids = ids.flatten()
	# lst = {}
	for (markerCorner, markerID) in zip(corners1, ids):
		# if markerID in [0,3]:
		# 	lst[markerID] = markerCorner.reshape((4, 2))
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
		cv2.line(image, topLeft, topRight, (0, 255, 0), 1)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 1)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 1)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 1)
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(image, (cX, cY), 1, (0, 0, 255), -1)
		cv2.putText(image, str(markerID),
            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1)
		print("ID: {}".format(markerID))
# def void():
# 	return
# cv2.namedWindow('hello')
# cv2.createTrackbar('Threshold: ', 'hello', 10, 25, void)
# cv2.imshow("Image", image)
# cv2.waitKey(0)

def slope(p1, p2):
	print(p1, p2, 'point')
	return (int(p2[1])-int(p1[1]))/(int(p2[0])-int(p1[0]))

def distance(p1, p2):
	return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

tl1, tr1, br1, bl1 = corners1[0].reshape((4, 2))
dist = np.sqrt((tr1[0]-tl1[0])**2 + (tr1[1]-tl1[1])**2)
print(dist, '###')
scale = 48/dist

tl2, tr2, br2, bl2 = corners1[1].reshape((4, 2))
dist = np.sqrt((tr2[0]-tl2[0])**2 + (tr2[1]-tl2[1])**2)
print(dist, '###')
print(distance(tl1, tr1), distance(tr1, br1))
print(distance(tl2, tr2), distance(tr2, br2))
# exit()
cx1, cy1 = (tl1[0]+br1[0])/2, (tl1[1]+br1[1])/2
cx2, cy2 = (tl2[0]+br2[0])/2, (tl2[1]+br2[1])/2
# print(cx1, cy1, cx2, cy2)
cv2.line(image, (int(br1[0]), int(br1[1])), (int(tr2[0]), int(tr2[1])), (0, 0, 255), 1)
cv2.imwrite('hello.jpg', image)
dist = np.sqrt((tr2[0]-br1[0])**2 + (tr2[1]-br1[1])**2)
# dist = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2)
print(dist, 'px')
print(dist*scale, 'mm')
# print(slope(tl1, tr1), slope(tl2, tr2), slope((cx1,cy1), (cx2,cy2)))
exit()
corner = 1
if corner:
	p1 = subpixel(image, corners1[0])[0]
	p2 = subpixel(image, corners1[3])[0]
	p11 = p1[0]
	p12 = p1[1]
	p21 = p2[0]
	p22 = p2[1]
	m1 = (p12[1] - p11[1]) / (p12[0] - p11[0])
	m2 = (p22[1] - p21[1]) / (p22[0] - p21[0])
else:
	p1 = subpixel(image, corners1[0])[0]
	p2 = subpixel(image, corners1[4])[0]
	p3 = subpixel(image, corners1[-2])[0]
	p11 = p1[0]
	p12 = p1[2]
	cx1 = (p11[0] + p12[0]) / 2
	cy1 = (p11[1] + p12[1]) / 2
	p21 = p2[0]
	p22 = p2[2]
	cx2 = (p21[0] + p22[0]) / 2
	cy2 = (p21[1] + p22[1]) / 2
	p31 = p3[0]
	p32 = p3[2]
	cx3 = (p31[0] + p32[0]) / 2
	cy3 = (p31[1] + p32[1]) / 2
	m1 = (cy3 - cy1) / (cx3 -cx1)
	m2 = (cy2 - cy1) / (cx2 -cx1)

angle = abs((m2 - m1) / (1 + m1 * m2))
import math
print(p11, p12, p21, p22)
print(m1, m2)
if angle==0:
	print(f"Offset: {abs(p11[1] - p21[1])}")
print(f"Angle: {math.degrees(angle)}")