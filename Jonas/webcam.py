import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import face_recognition
import cv2
import time


cap = cv2.VideoCapture(0)
model=tf.keras.models.load_model('models/128,64,64,5.model')

start=time.time()

while True:
    
    ret, img = cap.read()
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    face_locations = face_recognition.face_locations(img)
    if (np.shape(face_locations)[0])==0:
        cv2.imshow('Input', img)
        continue
    faces=np.empty((np.shape(face_locations)[0],48,48))
    
    
    for idx,face_coords in enumerate(face_locations):       
        face=img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
        #square crop still missing, there might be some stretching right now
        face_48=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
        faces[idx]=cv2.cvtColor(face_48, cv2.COLOR_BGR2GRAY)
        
#        img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]=20+0.9*img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
        img[face_coords[0]:face_coords[2],face_coords[3],1]=200
        img[face_coords[0]:face_coords[2],face_coords[1],1]=200
        img[face_coords[0],face_coords[3]:face_coords[1],1]=200
        img[face_coords[2],face_coords[3]:face_coords[1],1]=200
        
    model_faces=faces.reshape(np.shape(face_locations)[0],48,48,1) #adjust input shape
    model_faces=tf.keras.utils.normalize(model_faces,axis=1)

    predictions=model.predict(model_faces)
    
    end=time.time()
    fps=1/(end-start)
    start=time.time()
    
    emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    
    for idx, face in enumerate(faces):
            plt.imshow(face,cmap='gray')
            plt.show()
            print("FPS:",'%.1f' % fps)
            emotion=6
            high_score=0.0
            for i in range(7):
                if predictions[idx,i]>high_score:
                    high_score=predictions[idx,i]
                    emotion=i
                print(emotions[i],':',predictions[idx,i])
            print(emotions[emotion].upper())
    
    cv2.imshow('Input', img)
    
    

    c = cv2.waitKey(1)
    if c == 32: #space bar
        break

cap.release()
 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)