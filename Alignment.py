import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()