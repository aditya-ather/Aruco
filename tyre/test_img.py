import cv2
import numpy as np

BINARY_THRESHOLD = 127
CONTOUR_ASPECT_RATIO = 3
CONTOUR_EXTENT = 0.1
STRIP_COUNT = 2
CROP_IMAGE = False
CROP_WINDOW_X = 100
CROP_WINDOW_Y = 100
CROP_WINDOW_HEIGHT = 500
CROP_WINDOW_WIDTH = 1500
COLOR_DISTANCE_TOLERANCE = 50000
TYRE_VARIANTS = {
    'variant1': [(255, 255, 255), (255, 255, 255), (0, 255, 255)],
    'variant2': [(0, 255, 255), (0, 255, 255), (0, 0, 255)],
}

class Strip:
    def __init__(self, contour=None, rect=None):
        self.contour = contour
        self.rect = rect
    
    def center(self):
        M = cv2.moments(self.contour)
        cY = int(M["m01"] / M["m00"])
        cX = int(M["m10"] / M["m00"])
        return (cX, cY)

def find_contours(gray):
    _, thresh = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    gray = cv2.GaussianBlur(thresh, (5, 5), 0)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETRIEVE ONLY PARENT
    return contours

def filter_contours(contours):
    strips = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        extent = cv2.contourArea(cnt)/(w*h)
        if aspect_ratio > CONTOUR_ASPECT_RATIO and extent > CONTOUR_EXTENT:
            strip = Strip(contour=cnt, rect=(x, y, x+w, y+h))
            strips.append(strip)
    return strips

def filter_pixels(pixelpoints, gray):
    return np.array([point for point in pixelpoints if gray[point[0][1], point[0][0]] > BINARY_THRESHOLD])

def find_pixelpoints(contour, gray):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1) # "-1" means Filled solid white
    pixelpoints = cv2.findNonZero(mask)
    pixelpoints = filter_pixels(pixelpoints, gray)
    return pixelpoints

def select_strips(strips):
    for strip in strips:
        center = strip.center()
        selected_strips = []
        for strip2 in strips:
            x1, y1, x2, y2 = strip2.rect
            if x1 <= center[0] <= x2:
                selected_strips.append(strip2)
            if len(selected_strips) == STRIP_COUNT:
                break
        if len(selected_strips) == STRIP_COUNT:
            break
    if len(selected_strips) != STRIP_COUNT:
        print(f"Error: Could not find {STRIP_COUNT} strips")
        exit()
    return sorted(selected_strips, key=lambda strip: strip.center()[1])

def k_means(data, k=3, max_iterations=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iterations):
        labels = np.array([np.argmin(np.linalg.norm(point - centroids, axis=1)) for point in data])
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[i])
        new_centroids = np.array(new_centroids)
        if np.allclose(new_centroids, centroids, atol=1e-2):
            break
        centroids = new_centroids
    return centroids, labels

def extract_color(strip, gray, img, k=1, max_iterations=10):
    pixelpoints = find_pixelpoints(strip.contour, gray)
    masked_pixels = []
    for point in pixelpoints:
        x, y = point[0]
        pixel = img[y, x]
        masked_pixels.append(pixel)
    masked_pixels = np.array(masked_pixels).reshape(-1, 3)
    if k == 1:
        mean_col = np.mean(masked_pixels, axis=0)
        return tuple(int(pixel) for pixel in mean_col)
    centroids, labels = k_means(masked_pixels, k, max_iterations)
    largest_cluster_idx = np.argmax([np.sum(labels == i) for i in range(k)])
    return tuple(int(pixel) for pixel in centroids[largest_cluster_idx])

def find_variant(detected_pattern):
    metrics = {}
    for variant, pattern in TYRE_VARIANTS.items():
        if len(pattern) == STRIP_COUNT:
            distance = np.linalg.norm(np.array(pattern) - np.array(detected_pattern))
            metrics[variant] = distance
    if len(metrics) == 0:
        raise Exception("Error: No variant found")
    closest_idx = np.argmin(metrics.values())
    if min(metrics.values()) < COLOR_DISTANCE_TOLERANCE:
        return list(metrics.keys())[closest_idx]
    raise Exception("Error: No variant found")

if __name__ == "__main__":
    file = r"C:\Users\aditya.raj\Downloads\frame0.jpg"
    img = cv2.imread(file)
    try:
        if CROP_IMAGE:
            img = img[CROP_WINDOW_Y:CROP_WINDOW_Y+CROP_WINDOW_HEIGHT, CROP_WINDOW_X:CROP_WINDOW_X+CROP_WINDOW_WIDTH]
        img_copy = img.copy() ###### REMOVE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours = find_contours(gray)
        strips = filter_contours(contours)
        cv2.drawContours(img_copy, [strip.contour for strip in strips], -1, (255, 0, 0), 2) #### REMOVE
        if len(strips) < STRIP_COUNT:
            print(f"Error: Could not find {STRIP_COUNT} strips")
            cv2.imshow('img', cv2.resize(img_copy, (CROP_WINDOW_WIDTH, CROP_WINDOW_HEIGHT))) ### REMOVE
            cv2.waitKey(0) ### REMOVE
            cv2.destroyAllWindows() ### REMOVE
            exit()
        selected_strips = select_strips(strips)
        for strip in selected_strips: #### REMOVE
            x1, y1, x2, y2 = strip.rect #### REMOVE
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3) #### REMOVE
        pattern = []
        for strip in selected_strips:
            color = extract_color(strip, gray, img, k=1)
            pattern.append(color)
            cv2.rectangle(img_copy, (strip.rect[0], strip.rect[1]), (strip.rect[2], strip.rect[3]), color, -1) #### REMOVE
        print(pattern)
        # variant = find_variant(pattern)
        # print(variant)
        cv2.imshow('img', cv2.resize(img_copy, (CROP_WINDOW_WIDTH, CROP_WINDOW_HEIGHT))) ### REMOVE

        height, width = (300, STRIP_COUNT*50) #### REMOVE
        section_width = width // STRIP_COUNT #### REMOVE
        pattern_img = np.zeros((height, width, 3), dtype=np.uint8) #### REMOVE
        for i in range(STRIP_COUNT):
            pattern_img[:, i*section_width:(i+1)*section_width] = extract_color(selected_strips[i], gray, img) #### REMOVE
        cv2.imshow('Pattern', pattern_img) #### REMOVE
        cv2.imwrite('pattern_yellow.jpg', pattern_img) #### REMOVE
        cv2.imwrite('tyre_segment.jpg', img_copy) ### REMOVE
    except:
        pass
    cv2.waitKey(0) ### REMOVE
    cv2.destroyAllWindows() ### REMOVE


