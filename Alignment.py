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

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detection is not None:
            for detection in results.detections:
                #Ojo 1
                x1 = int(detection.location_data.relative_keypoints[0].x*width)
                y1 = int(detection.location_data.relative_keypoints[0].y*height)

                # Ojo 2
                x2 = int(detection.location_data.relative_keypoints[1].x * width)
                y2 = int(detection.location_data.relative_keypoints[1].y * height)

                #Visualizar datos
                cv2.putText(frame, "Ojo1", (x1 - 60, y1), 1, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, "Ojo2", (x2 + 10, y2), 1, 1.5, (0, 128, 255), 2)

                #Lados del triangulo
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (x1, y1), (x2, y1), (211, 0, 148), 2)
                cv2.line(frame, (x2, y2), (x2, y1), (0, 128, 255), 2)

                #Circulos en cada uno de los vertices del triangulo
                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 128, 255), -1)


        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()