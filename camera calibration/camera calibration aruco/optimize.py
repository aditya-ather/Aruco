import os
import cv2
from cv2 import aruco
import numpy as np
import scipy

calibrate_camera = True
calib_imgs_path = "C:/Users/aditya.raj/Desktop/ergometrics\data/calibration_images/54mp/focus5_8/"
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
arucoParams = aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementMinAccuracy = 1e-10
detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

img_list = []
calib_fnms = os.listdir(calib_imgs_path)
for idx, fn in enumerate(calib_fnms):
    img = cv2.imread(calib_imgs_path+fn)
    img_list.append( img )
    h, w, c = img.shape
counter, corners_list, id_list = [], [], []
first = True
for im in (img_list):
    img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img_gray)
    if first == True:
        corners_list = corners
        id_list = ids
        first = False
    else:
        try:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        except:
            print('fail')
            continue
    counter.append(len(ids))
counter = np.array(counter)

def calibrate(x, corners_list, id_list, counter, img_gray):
    markerLength=x[0]
    markerSeparation=x[1]
    board = aruco.GridBoard((7, 14), markerLength, markerSeparation, aruco_dict)
    ret, mtx, dist, rvecs, tvecs, _, _, perview = aruco.calibrateCameraArucoExtended(corners_list, id_list, counter, board, img_gray.shape, None, None)#, perViewErrors=perview)
    return ret

x0=[22.7289, 19.00] # initial values daal idhar markerlength aur separation ka
res=scipy.optimize.minimize(calibrate, x0, args=(corners_list, id_list, counter, img_gray), method='L-BFGS-B', options={'disp':True, 'eps':1e-3}, bounds=[(22.5,22.9), (18.9, 19.1), (3.7, 4)], tol=1e-5)
print(res)
