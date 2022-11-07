import cv2 as cv
import os
import numpy as np

i = '4'

# dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database'
dataPath = 'C:/Users/agusv/PycharmProjects/Reconocimiento_facial_de_pacientes/Database_' + i

peopleList = os.listdir(dataPath)
print('people list: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Reading images')

    for fileName in os.listdir(personPath):
        print('Faces: ', nameDir + '/' + fileName)
        labels.append(label)

        rostro = cv.imread(personPath + '/' + fileName, 0)
        image = cv.resize(rostro, (720, 720), interpolation=cv.INTER_CUBIC)
        facesData.append(image)
        #####################################################
        cv.imshow('image', image)
        cv.waitKey(10)
        #####################################################
    label = label + 1

cv.destroyAllWindows()

############################################
# print('labels = ', labels)
# print('Number of labels 0: ', np.count_nonzero(np.array(labels) == 0))
# print('Number of labels 0: ', np.count_nonzero(np.array(labels) == 1))
############################################
face_recognizer = cv.face.EigenFaceRecognizer_create()

print("Training...")
face_recognizer.train(facesData, np.array(labels))
print("Writing...")
face_recognizer.write('ModeloFaceData2022_' + i + '.xml')
print("Saved Model")
