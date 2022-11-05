from arcface import ArcFace
face_rec = ArcFace.ArcFace()
emb1 = face_rec.calc_emb("rostro_3.jpg")
print(emb1)