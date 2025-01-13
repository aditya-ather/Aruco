import cv2, os
# Open the video file
video_path = r'C:\Users\aditya.raj\Downloads\vid3.mp4'
path = r'tyre\pattern1'
os.makedirs(path, exist_ok=True)
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set the desired frame rate
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // fps)
# print(frame_interval)
# Read until the video is completed
i = 0
j = len(os.listdir(path))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if i % frame_interval == 0:
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        cv2.imwrite(f'tyre/pattern1/frame{j}.jpg', frame)
        j+=1

    i += 1
    # Press Q on keyboard to exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()