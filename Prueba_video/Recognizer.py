import cv2 as cv
import os

i = '4'

# dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database'
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i

imagePaths = os.listdir(dataPath)
print('imagePath = ', imagePaths)

print("Start reading Model")
face_recognizer = cv.face.EigenFaceRecognizer_create()
print("Model reading")
face_recognizer.read('ModeloFaceData2022_' + i + '.xml')
print("Model read")

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture(0)
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# print(faceClassif)

cut_param = 30700

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv.resize(rostro, (720, 720), interpolation=cv.INTER_CUBIC)
        # print("rostro")
        # print(rostro)

        result = face_recognizer.predict(rostro)
        # result1 = face_recognizer.predict_label(rostro)
        print("result")
        print(result)

        cv.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv.LINE_AA)

        if result[1] < cut_param:
            result_ = result[0]
            print("result_", result_)
            image_paths_result_ = imagePaths[result_]
            print("image_paths_result_", image_paths_result_)
            cv.putText(frame, '{}'.format(image_paths_result_), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv.putText(frame, 'Unknown', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

cap.realease()
cv.destroyAllWindows()
