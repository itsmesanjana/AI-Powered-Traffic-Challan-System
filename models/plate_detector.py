from __future__ import annotations

import os
import cv2
import numpy as np

def detect_number_plate(image):
    """
    Plate detection using traditional computer vision methods.
    Returns: (x, y, w, h) or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Try Haar Cascade first
    for cascade_path in [
        "haarcascade_license_plate_rus_16stages.xml",
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    ]:
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(plates) > 0:
                    # Sort by area
                    plates = sorted(plates, key=lambda p: p[2] * p[3], reverse=True)
                    return tuple(plates[0])

    # 2. Contour detection fallback
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    possible_plates = []

    img_h, img_w = image.shape[:2]
    img_cx = img_w / 2

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        aspect_ratio = w / float(h)
        box_cx = x + w / 2

        # Plate is usually rectangular and positioned near the horizontal center
        if (
            area > 1000
            and area < 40000
            and aspect_ratio >= 1.0
            and aspect_ratio <= 6.0
            and w > 60
            and h > 20
            and abs(box_cx - img_cx) < img_w * 0.40
        ):
            possible_plates.append((x, y, w, h, area))

    if not possible_plates:
        h_img, w_img = image.shape[:2]
        fallback_w = w_img // 2
        fallback_h = h_img // 3
        fallback_x = w_img // 4
        fallback_y = h_img - fallback_h
        return (fallback_x, fallback_y, fallback_w, fallback_h)

    # Choose best candidate: lowest in the image (max y coordinate) to avoid headlights
    possible_plates = sorted(possible_plates, key=lambda x: x[1], reverse=True)
    x, y, w, h, _ = possible_plates[0]
    return (x, y, w, h)


def draw_plate_box(image, bbox):
    output = image.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output
