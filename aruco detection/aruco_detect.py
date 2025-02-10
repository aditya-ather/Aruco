import os
import cv2
import numpy as np
import pickle

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.useAruco3Detection = True
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementMinAccuracy = 1e-10
aruco_params.adaptiveThreshWinSizeStep = 1
aruco_params.adaptiveThreshConstant = 15
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def slope(p1, p2):
	print(p1, p2, 'point')
	return (int(p2[1])-int(p1[1]))/(int(p2[0])-int(p1[0]))

def distance(p1, p2):
	return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle_between_vectors(v1, v2):
    try:
        v1 = v1.reshape((3,1))
        v2 = v2.reshape((3,1))
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = sum(unit_vector_1[i]*unit_vector_2[i] for i in range(3))
        # dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        # print(angle)
        return np.degrees(angle[0])
    except Exception as e:
         print(v1, v2)
         raise e


def subpixel(image, corners):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
markerLength = 28.5 #mm
cube = np.array([
        [0, 0, 0],
        [markerLength, 0, 0],
        [markerLength, markerLength, 0],
        [0, markerLength, 0],
        [0, 0, -markerLength],
        [markerLength, 0, -markerLength],
        [markerLength, markerLength, -markerLength],
        [0, markerLength, -markerLength]
    ], dtype=np.float32)
cube[:, :2] -= markerLength/2
cube[:, 2] *= -1


with open((r'C:\Users\aditya.raj\Desktop\ergometrics\src\calibration.pkl'), 'rb') as f:
    loadeddict = pickle.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeffs')
mtx = np.array(mtx)
dist = np.array(dist)

objPoints = np.array([[-markerLength / 2, markerLength / 2, 0],
                    [markerLength / 2, markerLength / 2, 0],
                    [markerLength / 2, -markerLength / 2, 0],
                    [-markerLength / 2, -markerLength / 2, 0]], dtype=np.float32)
lengths= []
for j in range(6):
	print(j)
	# a='img000{}.jpg'.format(i if i>=10 else f'0{i}')
	# if a not in os.listdir(r"C:\Users\aditya.raj\Desktop\ergometrics\data\logs"):
	# 	continue
	# frame = cv2.imread(r"C:\Users\aditya.raj\Desktop\ergometrics\data\logs\{}".format(a))
	frame = cv2.imread("C:/Users/aditya.raj/Desktop/ergometrics/data/raw/"+str(j)+".jpg")
	h,  w = frame.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	# frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
	# frame = cv2.resize(frame, (1400, 1088))
	(corners1, ids, rejected) = detector.detectMarkers(frame)
	if len(corners1)==0:
		# cv2.imshow('frame', cv2.resize(frame, (1080, 720)))
		# cv2.waitKey(0)
		print('no corners:', j)
		continue

	if len(corners1) > 0:
		# rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners1, 48, mtx, dist)
		rvecs, tvecs = [], []
		for corner in corners1:
			# If corners detected accurately but cube is displaced, intrinsic is wrong.
			_, rvec, tvec = cv2.solvePnP(objPoints, corner, mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
			# rvec, tvec = cv2.solvePnPRefineLM(objPoints, corner, mtx, dist, rvec, tvec, (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER, 3000000, 1e-10))
			rvec, tvec = cv2.solvePnPRefineVVS(objPoints, corner, mtx, dist, rvec, tvec, (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER, 3000000, 1e-10), 1)
			rvecs.append(rvec)
			tvecs.append(tvec)
		for rvec,tvec in zip(rvecs, tvecs):
			imgpts, _ = cv2.projectPoints(cube, rvec, tvec, mtx, dist)
			imgpts = np.int32(imgpts).reshape(-1,2)
			# print(imgpts.shape)
			cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
			cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)
			for i in range(4):
				cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (255, 0, 0), 1)
		ids = ids.flatten() 
		for (markerCorner, markerID) in zip(corners1, ids):
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 1)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 1)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 1)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 1)
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
	for i in range(len(tvecs)-1):
		t1, t2 = tvecs[i], tvecs[i+1]
		t1_2d = tuple(cv2.projectPoints(t1, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0].ravel().astype(int))
		t2_2d = tuple(cv2.projectPoints(t2, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0].ravel().astype(int))
		cv2.circle(frame, t1_2d, 1, (0, 0, 255), -1)
		cv2.circle(frame, t2_2d, 1, (0, 0, 255), -1)
		cv2.line(frame, t1_2d, t2_2d, (0,255,0), 1)
		a = np.linalg.norm(t1-t2)
		print(a)
		# if 200<a<400:
		lengths.append(a)
		cv2.putText(frame, f'{a:.1f}', ((t1_2d[0]+t2_2d[0])//2, (t1_2d[1]+t2_2d[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 0, 255), 2)
        
		n1, n2 = (cv2.Rodrigues(r)[0][:,2] for r in rvecs[i:i+2])
		angle1 = angle_between_vectors(n1, n2)
		print(angle1, 'deg')
		cv2.putText(frame, f'{angle1:.1f} deg', (-500+(t1_2d[0]+t2_2d[0])//2, (t1_2d[1]+t2_2d[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 0, 0), 2)

	# cv2.imshow(f"{i}", cv2.resize(frame, (1080, 720)))
	cv2.imwrite(f'output{j}.jpg', frame)	# cv2.waitKey(0)

# if cv2.waitKey(1) == ord('q'):
print(lengths)
print(np.median(lengths), 'mm')
print(np.mean(lengths), 'mm')
hist, bin_edges = np.histogram(lengths, bins=50)
max_bin_index = np.argmax(hist)
print(f'Top bin range: {bin_edges[max_bin_index]} - {bin_edges[max_bin_index + 1]} mm')