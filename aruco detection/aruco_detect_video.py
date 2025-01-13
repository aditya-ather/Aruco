import pickle
import cv2
import time

import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
cam = cv2.VideoCapture(1)

def slope(p1, p2):
	print(p1, p2, 'point')
	return (int(p2[1])-int(p1[1]))/(int(p2[0])-int(p1[0]))

def distance(p1, p2):
	return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
# cam = cv2.VideoCapture('http://172.19.66.48:8080/video')

with open(('aruco detection/calibration.pkl'), 'rb') as f:
    loadeddict = pickle.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

while True:
    ret, frame = cam.read()
    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # frame = cv2.resize(frame, (1400, 1088))
    (corners1, ids, rejected) = detector.detectMarkers(frame)

    if len(corners1) > 2:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners1, 48, mtx, dist)
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
            cv2.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
            cv2.putText(frame, str(markerID), (bottomLeft[0], bottomLeft[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # print("ID: {}".format(markerID))

        tl1, tr1, br1, bl1 = corners1[0].reshape((4, 2))
        tl2, tr2, br2, bl2 = corners1[1].reshape((4, 2))
        tl3, tr3, br3, bl3 = corners1[2].reshape((4, 2))
        # dist = np.sqrt((tr2[0]-tl2[0])**2 + (tr2[1]-tl2[1])**2)
        cx1, cy1 = (tl1[0]+br1[0])/2, (tl1[1]+br1[1])/2
        cx2, cy2 = (tl2[0]+br2[0])/2, (tl2[1]+br2[1])/2
        cx3, cy3 = (tl3[0]+br3[0])/2, (tl3[1]+br3[1])/2
        # dist = np.sqrt((tr2[0]-br1[0])**2 + (tr2[1]-br1[1])**2)
        a = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2)
        b = np.sqrt((cy2-cy3)**2 + (cx2-cx3)**2)
        c = np.sqrt((cy3-cy1)**2 + (cx3-cx1)**2)
        
        t1, t2, t3 = tvecs[0][0], tvecs[1][0], tvecs[2][0]
        t1_2d = tuple(cv2.projectPoints(t1, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0].ravel().astype(int))
        t2_2d = tuple(cv2.projectPoints(t2, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0].ravel().astype(int))
        t3_2d = tuple(cv2.projectPoints(t3, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0].ravel().astype(int))

        cv2.line(frame, t1_2d, t2_2d, (0,255,0), 2)
        cv2.line(frame, t3_2d, t2_2d, (0,255,0), 2)
        cv2.line(frame, t1_2d, t3_2d, (0,255,0), 2)

        a = np.linalg.norm(t1-t2)
        b = np.linalg.norm(t3-t2)
        c = np.linalg.norm(t1-t3)
        print(a, b, c)

        # dist = np.sqrt((tr1[0]-tl1[0])**2 + (tr1[1]-tl1[1])**2)
        scale = 298/max(a,b,c)
        print(a*scale, b*scale, c*scale, 'mm')
        # cv2.line(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 0, 255), 1)
        # cv2.line(frame, (int(cx1), int(cy1)), (int(cx3), int(cy3)), (0, 0, 255), 1)
        # cv2.line(frame, (int(cx3), int(cy3)), (int(cx2), int(cy2)), (0, 0, 255), 1)
        # time.sleep(0.5)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break



cam.release()
cv2.destroyAllWindows()