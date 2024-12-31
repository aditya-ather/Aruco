import cv2
import time

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
cam = cv2.VideoCapture(1)
# cam = cv2.VideoCapture('http://172.19.66.48:8080/video')

while True:
    ret, frame = cam.read()
    # frame = cv2.resize(frame, (1400, 1088))
    (corners, ids, rejected) = detector.detectMarkers(frame)

    if len(corners) > 0:
        ids = ids.flatten() 
        for (markerCorner, markerID) in zip(corners, ids):
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
            print("ID: {}".format(markerID))

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()