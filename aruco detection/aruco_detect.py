import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
image = cv2.imread(r"C:\Users\aditya.raj\Desktop\project_aruco\calibresult.jpg")
image = cv2.resize(image, (1300, 400))
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
(corners1, ids, rejected) = detector.detectMarkers(image)

if len(corners1) > 0:
	ids = ids.flatten()
	for (markerCorner, markerID) in zip(corners1, ids):
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
		cv2.circle(image, (cX, cY), 2, (0, 0, 255), -1)
		cv2.putText(image, str(markerID),
            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1)
		print("ID: {}".format(markerID))
cv2.imshow("Image", image)
cv2.waitKey(0)


corner = 0
if corner:
	# For corner as reference
	p11 = corners1[0].reshape((4,2))[0]
	p12 = corners1[0].reshape((4,2))[1]
	p21 = corners1[-2].reshape((4,2))[0]
	p22 = corners1[-2].reshape((4,2))[1]
	m1 = (p12[1] - p11[1]) / (p12[0] - p11[0])
	m2 = (p22[1] - p21[1]) / (p22[0] - p21[0])
else:
	# for center as reference
	p11 = corners1[0].reshape((4,2))[0]
	p12 = corners1[0].reshape((4,2))[2]
	cx1 = (p11[0] + p12[0]) / 2
	cy1 = (p11[1] + p12[1]) / 2
	p21 = corners1[-1].reshape((4,2))[0]
	p22 = corners1[-1].reshape((4,2))[2]
	cx2 = (p21[0] + p22[0]) / 2
	cy2 = (p21[1] + p22[1]) / 2
	p31 = corners1[-2].reshape((4,2))[0]
	p32 = corners1[-2].reshape((4,2))[2]
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