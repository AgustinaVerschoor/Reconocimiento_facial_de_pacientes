import cv2 as cv
import os

# dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database'
# dataPath = '/Users/maxiadaro/sandbox/austral/visiar_Agus/Reconocimiento_facial_de_pacientes/Database'
#
# imagePaths = os.listdir(dataPath)
# print('imagePath = ', imagePaths)

# face_recognizer = cv.face.EigenFaceRecognizer_create()

# face_recognizer.read('ModeloFaceData2022.xml')
# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# while True:
#     ret, frame = cap.read()
#     if ret == False: break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     auxFrame = gray.copy()
#
#     faces = faceClassif.detectMultiScale(gray, 1.3, 5)
#
#     for (x, y, w, h) in faces:
#         rostro = auxFrame[y:y + h, x:x + w]
#         rostro = cv.resize(rostro, (720, 720), interpolation=cv.INTER_CUBIC)
#         result = face_recognizer.predict(rostro)
#
#         cv.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv.LINE_AA)
#
#         if result[1] < 5700:
#             cv.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         else:
#             cv.putText(frame, 'Unknown', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
#             cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     cv.imshow('frame', frame)
#     k = cv.waitKey(1)
#     if k == 27:
#         break

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
cap.realease()
# cv.destroyAllWindows()
