"""
This code assumes that images used for calibration are of the same arUco marker board provided with code

"""

import pickle
import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path
from tqdm import tqdm

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
# calibrate_camera = True
calibrate_camera = True

# Set path to the images
calib_imgs_path = root.joinpath("images webcam")

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )

#Provide length of the marker's side
markerLength = 3.8  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard((4, 5), markerLength, markerSeparation, aruco_dict)
# img = board.generateImage((1080,680))
# cv2.imshow("aruco", img)
# cv2.waitKey(0)

arucoParams = aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    for idx, fn in enumerate(calib_fnms):
        img = cv2.imread( str(root.joinpath(fn) ))
        img_list.append( img )
        h, w, c = img.shape

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
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
    print('Found {} unique markers'.format(len(np.unique(ids))))

    counter = np.array(counter)
    print ("Calibrating camera ...")
    #mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    # print("Camera matrix is \n", mtx, "\n And is stored in calibration.pkl file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open(root.joinpath("calibration.pkl"), "wb") as f:
        pickle.dump(data, f)
    print('Calibration done.')
    print("Total reprojection error: ", ret)
    print(data)

else:
    camera = cv2.VideoCapture(1)
    # camera = cv2.VideoCapture('https://172.19.66.48:8080/video')
    ret, img = camera.read()

    with open(root.joinpath('calibration.pkl'), 'rb') as f:
        loadeddict = pickle.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    # mtx = np.array([[626.0, 0.0, 324.0],[0.0, 628.0, 258.0],[0.0, 0.0, 1.0]])
    dist = np.array(dist)
    # dist = np.array([[0.068, -0.16, 0.001, 0.0, -0.027]])

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    cube = np.array([
        [0, 0, 0],
        [markerLength, 0, 0],
        [markerLength, markerLength, 0],
        [0, markerLength, 0],
        [0, 0, -markerLength],
        [markerLength, 0, -markerLength],
        [markerLength, markerLength, -markerLength],
        [0, markerLength, -markerLength]
    ], dtype=np.float32) * 2

    offset = np.zeros((8,3))
    offset[:, 2] = -20
    # cube += offset

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = detector.detectMarkers(dst)

        if corners:
            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist, None, None) # For a board
            
            if ret != 0:
                # img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                # img_aruco = cv2.drawFrameAxes(img_aruco, newcameramtx, dist, rvec, tvec, 10, 5)    # axis length 10 can be changed according to your requirement
                
                imgpts, _ = cv2.projectPoints(cube, rvec, tvec, newcameramtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2)
                cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
                cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)
                for i in range(4):
                    cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]), (255, 0, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('s') and len(corners)>3:
                use_corner = 1
                if use_corner:
                    # For corner as reference
                    p1 = [corners[i] for i in range(len(corners)) if ids[i]==0][0]
                    p2 = [corners[i] for i in range(len(corners)) if ids[i]==3][0]
                    p11 = p1.reshape((4,2))[0]
                    p12 = p1.reshape((4,2))[1]
                    p21 = p2.reshape((4,2))[0]
                    p22 = p2.reshape((4,2))[1]
                    m1 = (p12[1] - p11[1]) / (p12[0] - p11[0])
                    m2 = (p22[1] - p21[1]) / (p22[0] - p21[0])
                else:
                    # for center as reference
                    p1 = [corners[i] for i in range(len(corners)) if ids[i]==0][0]
                    p2 = [corners[i] for i in range(len(corners)) if ids[i]==2][0]
                    p3 = [corners[i] for i in range(len(corners)) if ids[i]==3][0]

                    p11 = p1.reshape((4,2))[0]
                    p12 = p1.reshape((4,2))[2]
                    cx1 = (p11[0] + p12[0]) / 2
                    cy1 = (p11[1] + p12[1]) / 2
                    p21 = p2.reshape((4,2))[0]
                    p22 = p2.reshape((4,2))[2]
                    cx2 = (p21[0] + p22[0]) / 2
                    cy2 = (p21[1] + p22[1]) / 2
                    p31 = p3.reshape((4,2))[0]
                    p32 = p3.reshape((4,2))[2]
                    cx3 = (p31[0] + p32[0]) / 2
                    cy3 = (p31[1] + p32[1]) / 2
                    m1 = (cy3 - cy1) / (cx3 -cx1)
                    m2 = (cy2 - cy1) / (cx2 -cx1)

                angle = abs((m2 - m1) / (1 + m1 * m2))
                import math
                if use_corner:
                    print(p11, p12, p21, p22)
                else:
                    print([int(cx1), int(cy1)], [int(cx2), int(cy2)], [int(cx3), int(cy3)])
                print(m1, m2)
                if angle==0:
                    print(f"Offset: {abs(p11[1] - p21[1])}")
                print(f"Angle: {math.degrees(angle)}")
        cv2.imshow("hbfjd", img_aruco)
        # cv2.imshow("World co-ordinate frame axes", cv2.resize((img_aruco), (640, 320)))


        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

cv2.destroyAllWindows()