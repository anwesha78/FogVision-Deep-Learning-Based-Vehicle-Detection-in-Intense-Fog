import cv2
import numpy as np

def dehaze_image(image):

    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge channels back
    merged = cv2.merge((cl, a, b))

    # Convert back to BGR
    dehazed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return dehazed