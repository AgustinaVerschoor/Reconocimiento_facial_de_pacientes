# from deepface import DeepFace
#
# #Metodo que compara 2 imagenes, una seria la registrada en la base de datos, otra la que capta en la webcam
# obj = DeepFace.verify("Sofi.jpeg", "Sofi_playa.jpeg"
#           , model_name = 'ArcFace', detector_backend = 'retinaface')
# print(obj["verified"])
#############################################################
# from deepface.commons import functions
#
# img1_path = "Sofi.jpeg"
# img2_path = "Sofi_playa.jpeg"
#
# img1 = functions.preprocess_face(img1_path, target_size=(112, 112))
# img2 = functions.preprocess_face(img2_path, target_size=(112, 112))
#
# img1_embedding = model.predict(img1)[0]
# img2_embedding = model.predict(img2)[0]

#############################################################

from deepface import DeepFace

DeepFace.stream("../Database", model_name ="DeepFace")


