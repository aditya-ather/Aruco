import cv2
import numpy as np

BINARY_THRESHOLD = 127
CONTOUR_ASPECT_RATIO = 3
CONTOUR_EXTENT = 0.1
STRIP_COUNT = 2
CROP_IMAGE = False
CROP_WINDOW_X = 100
CROP_WINDOW_Y = 100
CROP_WINDOW_HEIGHT = 200
CROP_WINDOW_WIDTH = 600
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
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("Error: No contours found")
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
    if len(strips) == 0:
        raise Exception("Error: No strips found")
    return strips

def filter_pixels(pixelpoints, gray):
    return np.array([point for point in pixelpoints if gray[point[0][1], point[0][0]] > BINARY_THRESHOLD])

def find_pixelpoints(contour, gray):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    pixelpoints = cv2.findNonZero(mask)
    pixelpoints = filter_pixels(pixelpoints, gray)
    return pixelpoints

def select_strips(strips):
    if len(strips) < STRIP_COUNT:
        raise Exception(f"Error: Found only {len(strips)} strips")
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
        raise Exception(f"Error: Found only {len(strips)} strips")
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

def main():
    # cam = cv2.VideoCapture(0)
    cam = cv2.VideoCapture(r'C:\Users\aditya.raj\Downloads\vid1.mp4')
    while True:
        _, img = cam.read()
        try:
            if CROP_IMAGE:
                img = img[CROP_WINDOW_Y:CROP_WINDOW_Y+CROP_WINDOW_HEIGHT, CROP_WINDOW_X:CROP_WINDOW_X+CROP_WINDOW_WIDTH]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_copy = img.copy() ###### REMOVE
            contours = find_contours(gray)
            cv2.drawContours(img_copy, contours, 0, (0, 0, 255), 10)
            strips = filter_contours(contours)
            cv2.drawContours(img_copy, [strip.contour for strip in strips], -1, (255, 0, 0), 2) #### REMOVE
            selected_strips = select_strips(strips)
            pattern = []
            for strip in selected_strips:
                color = extract_color(strip, gray, img, k=1)
                cv2.rectangle(img_copy, (strip.rect[0], strip.rect[1]), (strip.rect[2], strip.rect[3]), color, -1) #### REMOVE
                pattern.append(color)
            print(pattern)
            variant = find_variant(pattern)
            print(variant)
            cv2.imshow('img', cv2.resize(img_copy, (700, 700))) ### REMOVE
            height, width = (300, STRIP_COUNT*50) #### REMOVE
            section_width = width // STRIP_COUNT #### REMOVE
            pattern_img = np.zeros((height, width, 3), dtype=np.uint8) #### REMOVE
            for i in range(STRIP_COUNT):
                pattern_img[i*section_width:(i+1)*section_width] = extract_color(selected_strips[i], gray, img) #### REMOVE
            cv2.imshow('Pattern', pattern_img) #### REMOVE
            cv2.imwrite('pattern.jpg', pattern_img) #### REMOVE
        except Exception as e:
            print(e)
            pass
        cv2.imshow("Camera", img_copy)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

