import cv2
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 512 # pixels
os.makedirs(f'aruco_markers/', exist_ok=True)
for marker_id in [3, 12, 15, 20, 27, 39, 49]:
    tag = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    cv2.imwrite(f'aruco_markers/{marker_id}.png', tag)
# cv2.imshow("ArUCo Tag", tag)
# cv2.waitKey(0)
print('end ')