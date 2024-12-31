import cv2
FOLDER = "checkerboards"
cam = cv2.VideoCapture(1)
i=0
while True:
    ret, frame = cam.read()
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite(f"{FOLDER}/{i}.jpg", frame)
        print(f"Image {i} saved")
        i+=1
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()