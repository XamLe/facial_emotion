import face_recognition
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image  

# Comment for testing purpose

img = cv2.imread('faces.jpg')

face_locations = face_recognition.face_locations(img)


for face_coords in face_locations:
    face=img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
    face_48=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
    face_48_gr = cv2.cvtColor(face_48, cv2.COLOR_BGR2GRAY)
    Image.fromarray(face_48_gr).show()

#Hallo this is a Comment
