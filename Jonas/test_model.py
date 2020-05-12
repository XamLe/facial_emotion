import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import face_recognition
import cv2


img = cv2.imread('faces.jpg') 

face_locations = face_recognition.face_locations(img)
faces=np.empty((np.shape(face_locations)[0],48,48))


for idx,face_coords in enumerate(face_locations):       
    face=img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
    #square crop still missing, there might be some stretching right now
    face_48=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
    faces[idx]=cv2.cvtColor(face_48, cv2.COLOR_BGR2GRAY)
    
    

model=tf.keras.models.load_model('facial_emotion.model')
faces=tf.keras.utils.normalize(faces,axis=1)

predictions=model.predict(faces.reshape(np.shape(face_locations)[0],48,48,1))


emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

for idx, face in enumerate(faces):
        plt.imshow(face,cmap='gray')
        plt.show()
        for i in range(7):
            print(emotions[i],':',predictions[idx,i])
        