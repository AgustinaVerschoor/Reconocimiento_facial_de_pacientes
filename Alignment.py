import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:

     while True:
          ret, frame = cap.read()
          if ret == False:
               break

          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_detection.process(frame_rgb)

          if results.detections is not None:
               for detection in results.detections:
                    # Ojo 1
                    x1 = int(detection.location_data.relative_keypoints[0].x * width)
                    y1 = int(detection.location_data.relative_keypoints[0].y * height)

                    # Ojo 2
                    x2 = int(detection.location_data.relative_keypoints[1].x * width)
                    y2 = int(detection.location_data.relative_keypoints[1].y * height)

p1 = np.array([x1, y1])
p2 = np.array([x2, y2])
p3 = np.array([x2, y1])

# Obtenemos las distancias de: d_eyes, l1
d_eyes = np.linalg.norm(p1 - p2)
l1 = np.linalg.norm(p1 - p3)

# Calcular el ángulo formado por d_eyes y l1
angle = degrees(acos(l1 / d_eyes))

# Determinar si el ángulo es positivo o negativo
if y1 < y2:
     angle = - angle
# Rotar la imagen de entrada, para alinear el rostro
M = cv2.getRotationMatrix2D((width // 2, height // 2), -angle, 1)
aligned_image = cv2.warpAffine(frame, M, (width, height))
cv2.imshow("Aligned_image", aligned_image)

# Visualizar datos
cv2.putText(frame, "Ojo1", (x1 - 60, y1), 1, 1.5, (0, 255, 0), 2)
cv2.putText(frame, "Ojo2", (x2 + 10, y2), 1, 1.5, (0, 128, 255), 2)
cv2.putText(frame, str(int(angle)), (x1 - 35, y1 + 15), 1, 1.2, (0, 255, 0), 2)

# Lados del triángulo
cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.line(frame, (x1, y1), (x2, y1), (211, 0, 148), 2)
cv2.line(frame, (x2, y2), (x2, y1), (0, 128, 255), 2)

# Círculos en cada uno de los vértices del triángulo
cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
cv2.circle(frame, (x2, y2), 5, (0, 128, 255), -1)
cv2.circle(frame, (x2, y1), 5, (211, 0, 148), -1)

# Detección facial 2
results2 = face_detection.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))

if results2.detections is not None:
    for detection in results2.detections:
        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)

        if xmin < 0 or ymin < 0:
            continue

        aligned_face = aligned_image[ymin: ymin + h, xmin: xmin + w]
        cv2.imshow("aligned_face", aligned_face)

cv2.imshow("Frame", frame)
k = cv2.waitKey(1) & 0xFF
# Detección facial 2
results2 = face_detection.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))

if results2.detections is not None:
    for detection in results2.detections:
        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
        w = int(detection.location_data.relative_bounding_box.width * width)
        h = int(detection.location_data.relative_bounding_box.height * height)

        if xmin < 0 or ymin < 0:
            continue

        aligned_face = aligned_image[ymin: ymin + h, xmin: xmin + w]
        cv2.imshow("aligned_face", aligned_face)

cv2.imshow("Frame", frame)
k = cv2.waitKey(1) & 0xFF
# if k == 27:
#     break



cap.release()
cv2.destroyAllWindows()