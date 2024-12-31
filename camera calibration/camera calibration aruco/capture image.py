import cv2, os
from pathlib import Path

# camera = cv2.VideoCapture(1)
camera = cv2.VideoCapture('https://172.19.66.48:8080/video')
ret, img = camera.read()


root = Path(__file__).parent.absolute()
folder = "images/"
path = root.joinpath(folder)
os.makedirs(path, exist_ok=True)
count = 1
while True:
    name = str(count)+".jpg"
    ret, img = camera.read()
    cv2.imshow("img", cv2.resize(img, (1280, 640)))


    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite(path.joinpath(name), img)
        print(f"Image {count} saved")
        count += 1
    if cv2.waitKey(1) == ord('x'):
        break