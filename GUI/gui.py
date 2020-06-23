import tkinter as tk
import cv2
import numpy as np
from mss import mss
from PIL import Image, ImageTk

from webcam import Webcam
from Input import Input_Image
from CNN import CNN
from screen_grab import Screengrab
from faces import Faces
from train_cnn import Train_CNN

from new_algorithm import Algorithm

root=tk.Tk()
root.title("Emotion Recognition")
root.minsize(400,400)
height=720
width=1280
root.geometry(str(width)+'x'+str(height))



spacing=10
relxspacing=spacing/width
relyspacing=spacing/height


#Faces
faces=Faces(root,_relx=1-relxspacing,_relwidth=(1-relxspacing*4)/3,_rely=relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='ne')

#Input Image
input=Input_Image(root,faces,_relx=relxspacing,_relwidth=(1-relxspacing*4)/3,_rely=relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='nw')

#Input Webcam
cam=Webcam(root,input,_relx=relxspacing,_relwidth=(1-relxspacing*4)/3,_rely=1-relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='sw')

#CNN
cnn=CNN(root,input,faces,_relx=2*relxspacing+(1-relxspacing*4)/3,_relwidth=(1-relxspacing*4)/3,_rely=relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='nw')

#Screengrab
screengrab=Screengrab(root,input,_relx=2*relxspacing+(1-relxspacing*4)/3,_relwidth=(1-relxspacing*4)/3,_rely=1-relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='sw')

#train cnn
# train_cnn=Train_CNN(root,_relx=1-relxspacing,_relwidth=(1-relxspacing*4)/3,_rely=1-relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='se')

#new algorithm
new_algorithm=Algorithm(root,input,faces,_relx=1-relxspacing,_relwidth=(1-relxspacing*4)/3,_rely=1-relyspacing,_relheight=(1-relyspacing*3)/2,_anchor='se')



root.mainloop()