import cv2
import numpy as np
import os

# Constants
A4_WIDTH_MM = 210  # A4 paper width in mm
A4_WIDTH_PX = 2480  # A4 paper width in pixels at 300 DPI
MARKER_FOLDER = 'aruco_markers'
# MARKER_SIZE_MM = 50  # Marker size in mm
for MARKER_SIZE_MM in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
    MARKER_SIZE_PX = int((MARKER_SIZE_MM / A4_WIDTH_MM) * A4_WIDTH_PX)
    # Load marker images
    marker_images = []
    files = os.listdir(MARKER_FOLDER)
    for i in range(len(files)):
        marker_path = os.path.join(MARKER_FOLDER, files[i])
        marker_image = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
        if marker_image is not None:
            marker_images.append(marker_image)

    # Resize marker images to fit the collage
    marker_images = [cv2.resize(img, (MARKER_SIZE_PX, MARKER_SIZE_PX)) for img in marker_images]

    # Constants for spacing and text
    GAP_SIZE_PX = (20 if MARKER_SIZE_MM > 20 else 100)  # Gap size in pixels
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2
    FONT_THICKNESS = 5
    TEXT_COLOR = (0, 0, 0)  # Black text

    # Adjust collage size to include gaps
    collage_height = (2 if MARKER_SIZE_MM > 25 else 1) * (MARKER_SIZE_PX + GAP_SIZE_PX + 2 * FONT_SCALE * 30)  # Height of two rows of markers
    collage_width = A4_WIDTH_PX  # Full width of A4 paper
    collage = np.ones((collage_height, collage_width), dtype=np.uint8) * 255  # White background

    # Place markers in collage with gaps and subtitles
    for idx, marker in enumerate(marker_images):
        row = idx // (A4_WIDTH_PX // (MARKER_SIZE_PX + GAP_SIZE_PX))
        col = idx % (A4_WIDTH_PX // (MARKER_SIZE_PX + GAP_SIZE_PX))
        x_offset = col * (MARKER_SIZE_PX + GAP_SIZE_PX) + GAP_SIZE_PX
        y_offset = row * (MARKER_SIZE_PX + GAP_SIZE_PX + 2 * FONT_SCALE * 30) + GAP_SIZE_PX + FONT_SCALE * 30
        collage[y_offset:y_offset + MARKER_SIZE_PX, x_offset:x_offset + MARKER_SIZE_PX] = marker
        
        # Add subtitle (ID) below the marker
        text = f'{files[idx].split(".")[0]}'
        text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = x_offset + (MARKER_SIZE_PX - text_size[0]) // 2
        text_y = y_offset + MARKER_SIZE_PX + text_size[1] + 5
        cv2.putText(collage, text, (text_x, text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        # Add subtitle (marker size) above the marker
        size_text = f'{MARKER_SIZE_MM}mm'
        size_text_size = cv2.getTextSize(size_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        size_text_x = x_offset + (MARKER_SIZE_PX - size_text_size[0]) // 2
        size_text_y = y_offset - 10
        cv2.putText(collage, size_text, (size_text_x, size_text_y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    # Save collage
    cv2.imwrite(f'collage_{MARKER_SIZE_MM}mm.png', collage)

    # cv2.imshow('Collage', collage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()