from deepface import DeepFace

# DeepFace.stream("../Database", model_name="DeepFace")
# DeepFace.detectFace("../Database/Maxi/rostro_50.jpg", db_path= "../Database")

result = DeepFace.verify(img1_path="../Database/Maxi/rostro_50.jpg",
                         img2_path="../Database/Maxi/rostro_251.jpg",
                         model_name="DeepFace",
                         align=False)
print(result)
