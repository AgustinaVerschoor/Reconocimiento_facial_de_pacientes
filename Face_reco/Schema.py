import cv2 as cv
import os
import imutils
from time import sleep

i = '1'
personName = 'Sofia_Tartara'
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i
personPath = dataPath + '/' + personName
##########################################
if not os.path.exists(personPath): #chequea si existe la carpeta. si no existe la crea
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)


cap = cv.VideoCapture(0)


faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

PHOTOS_TO_TAKE = 30

while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width = 640)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv.resize(rostro, (720, 720), interpolation = cv.INTER_CUBIC)
        cv.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)

    count = count + 1

    cv.imshow('frame', frame)

    if cv.waitKey(1) == 27 or count >= PHOTOS_TO_TAKE:
        break

cap.release()
cv.destroyAllWindows()
