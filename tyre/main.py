import cv2
import numpy as np

def detect(window):
    # Convert to HSV color space
    hsv = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)

    # Color ranges
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)

    lower_blue = (110, 50, 50)
    upper_blue = (130, 255, 255)

    lower_yellow = (20, 100, 100)
    upper_yellow = (30, 255, 255)

    lower_white = (0, 0, 200)
    upper_white = (180, 25, 255)

    lower_black = (0, 0, 0)
    upper_black = (180, 255, 30)

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours in each mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects_red = [cv2.boundingRect(cnt) for cnt in contours_red]
    rects_blue = [cv2.boundingRect(cnt) for cnt in contours_blue]
    rects_yellow = [cv2.boundingRect(cnt) for cnt in contours_yellow]
    rects_white = [cv2.boundingRect(cnt) for cnt in contours_white]
    rects_black = [cv2.boundingRect(cnt) for cnt in contours_black]

    # Draw bounding boxes around red contours
    area_red = 0
    for cnt in contours_red:
        x, y, w, h = cv2.boundingRect(cnt)
        area_red += w * h
        cv2.rectangle(window, (x, y), (x + w, y + h), (0, 0, 255), 1)

    area_yellow = 0
    for cnt in contours_yellow:
        x, y, w, h = cv2.boundingRect(cnt)
        area_yellow += w * h
        cv2.rectangle(window, (x, y), (x + w, y + h), (30, 255, 255), 1)

    area_blue = 0
    for cnt in contours_blue:
        x, y, w, h = cv2.boundingRect(cnt)
        area_blue += w * h
        cv2.rectangle(window, (x, y), (x + w, y + h), (255, 0, 0), 1)

    area_white = 0
    for cnt in contours_white:
        x, y, w, h = cv2.boundingRect(cnt)
        area_white += w * h
        cv2.rectangle(window, (x, y), (x + w, y + h), (255, 255, 255), 1)

    area_black = 0
    for cnt in contours_black:
        x, y, w, h = cv2.boundingRect(cnt)
        area_black += w * h
        cv2.rectangle(window, (x, y), (x + w, y + h), (0, 0, 0), 1)

    return window, {'red': rects_red if area_red > min_area else [], 'blue': rects_blue if area_blue > min_area else [], 'yellow': rects_yellow if area_yellow > min_area else [], 'white': rects_white if area_white > min_area else [], 'black': rects_black if area_black > min_area else []}

file = r"tyre\tyre images\IMG_20241220_124835.jpg"
img = cv2.imread(file)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

window_width = 600
window_height = 200

step_size = 20

min_area = 1000

height, width, _ = img.shape
center_x = width // 2

classes = {
    'variant1': ['white', 'white', 'yellow'],
    'variant2': ['yellow', 'yellow', 'red'],
    # 2: ['white', 'white', 'yellow'],
    # 3: ['white', 'white', 'yellow'],
}

for y in range(0, height - window_height + 1, step_size):
    window = img[y:y + window_height, center_x - window_width // 2:center_x + window_width // 2]
    window, rects = detect(window)
    for variant, color_order in classes.items():
        if all([color in rects for color in color_order]):
            if np.mean(rects[color_order[0]]) <= np.mean(rects[color_order[1]]) <= np.mean(rects[color_order[2]]):
                print(f'{variant} detected')
            elif np.mean(rects[color_order[2]]) <= np.mean(rects[color_order[1]]) <= np.mean(rects[color_order[0]]):
                print(f'{variant} detected')
    
    # edges = cv2.Canny(window, 100, 200)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Window', window)
    # gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(window, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Detected Colors', window)
    cv2.waitKey(0)

cv2.destroyAllWindows()