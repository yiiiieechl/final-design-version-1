# preprocess.py

import cv2

def preprocess_frame(frame):

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame
