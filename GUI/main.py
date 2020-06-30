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
from trainingsdata import Trainingsdata
from video import Video
from adaboost import Adaboost

from new_algorithm import Algorithm

root=tk.Tk()
root.title("Emotion Recognition")
root.minsize(400,400)
height=760
width=1280
root.geometry(str(width)+'x'+str(height))



spacing=10
relxspacing=spacing/width
relyspacing=spacing/height

grid_x=4
grid_y=3

relgridheight=(1-(grid_y+1)*relyspacing)/grid_y
relgridwidth=(1-(grid_x+1)*relxspacing)/grid_x



#Faces
faces=Faces(root,_relx=1-relxspacing,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=relyspacing,_relheight=2*(1-relyspacing*(grid_y+1))/(grid_y)+relyspacing,_anchor='ne')

#Input Image
input=Input_Image(root,faces,_relx=relxspacing,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=relyspacing,_relheight=(1-relyspacing*(grid_y+1))/(grid_y),_anchor='nw')
(grid_x+1)
#Input Webcam
cam=Webcam(root,input,_relx=relxspacing,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=1-relyspacing,_relheight=(1-relyspacing*(grid_y+1))/(grid_y),_anchor='sw')

#CNN
cnn=CNN(root,input,faces,_relx=2*relxspacing+(1-relxspacing*(grid_x+1))/grid_x,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=relyspacing,_relheight=(1-relyspacing*(grid_y+1))/(grid_y),_anchor='nw')

#AdaBoost
adaboost=Adaboost(root,input,faces,_relx=relxspacing,_relwidth=relgridwidth,_rely=relyspacing*2+relgridheight,_relheight=relgridheight,_anchor='nw')

#Screengrab
screengrab=Screengrab(root,input,_relx=2*relxspacing+(1-relxspacing*(grid_x+1))/grid_x,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=1-relyspacing,_relheight=(1-relyspacing*(grid_y+1))/(grid_y),_anchor='sw')

#train cnn
train_cnn=Train_CNN(root,_relx=1-relxspacing,_relwidth=(1-relxspacing*(grid_x+1))/grid_x,_rely=1-relyspacing,_relheight=(1-relyspacing*(grid_y+1))/(grid_y),_anchor='se')

video=Video(root,input,screengrab,cnn,adaboost,cam,_relx=relxspacing*3+2*relgridwidth,_relwidth=relgridwidth,_rely=relyspacing*3+2*relgridheight,_relheight=relgridheight,_anchor='nw')


trainingsdata=Trainingsdata(root,input,cnn,adaboost,_relx=relxspacing*3+2*relgridwidth,_relwidth=relgridwidth,_rely=relyspacing,_relheight=relgridheight,_anchor='nw')

#new algorithm
new_algorithm2=Algorithm(root,input,faces,_relx=relxspacing*2+relgridwidth,_relwidth=relgridwidth,_rely=relyspacing*2+relgridheight,_relheight=relgridheight,_anchor='nw')

#new algorithm
new_algorithm3=Algorithm(root,input,faces,_relx=relxspacing*3+2*relgridwidth,_relwidth=relgridwidth,_rely=relyspacing*2+relgridheight,_relheight=relgridheight,_anchor='nw')


# comparison=tk.Frame(root,bg='#ffc180',bd=5)
# comparison.place(relx=relxspacing*4+3*relgridwidth,relwidth=relgridwidth,rely=relyspacing*2+relgridheight,relheight=relgridheight,anchor='nw')
# faces_label=tk.Label(comparison,text = 'Comparison')
# faces_label.pack()



root.mainloop()
