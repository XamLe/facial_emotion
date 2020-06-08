import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import face_recognition
import os

global cap,camera,photo,framecap
Photo_taken=False
video_capture=False
def start_video_capture():
    global cap,camera,video_capture,frame
    if video_capture==False:
        video_capture=True
      
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width*0.5)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height*0.5)
        
        camera = tk.Label(frame,bg='#80c1ff')
        camera.place(relx=0.05,rely=0.05,relwidth=0.4,relheight=0.4)
        
        def show_frame():
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cv2image=cv2.resize(cv2image,(0,0),fx=0.25,fy=0.25,interpolation = cv2.INTER_AREA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            camera.imgtk = imgtk
            camera.configure(image=imgtk)
            camera.after(10, show_frame)
        
        show_frame()
        return cap
 
def stop_video_capture(): 
    global cap,camera,video_capture
    if video_capture==True:
        video_capture=False
    camera.destroy()
    cap.release()
    
def take_photo():
    global camera,photo,frame,cap,video_capture,Photo_taken,framecap
    if video_capture==True:
        if Photo_taken==False:
            photo=tk.Label(frame,bg='#80c1ff')
            photo.place(relx=0.95,rely=0.05,relwidth=0.4,relheight=0.4,anchor='ne')
            Photo_taken=True
        _, framecap = cap.read()
        framecap = cv2.flip(framecap, 1)
        cv2image = cv2.cvtColor(framecap, cv2.COLOR_BGR2RGBA)
        cv2image=cv2.resize(cv2image,(0,0),fx=0.25,fy=0.25,interpolation = cv2.INTER_AREA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        photo.imgtk = imgtk
        photo.configure(image=imgtk)
        
        
def Run_model():
    global framecap,model_name
    img=framecap
    face_locations = face_recognition.face_locations(img)
    faces=np.empty((np.shape(face_locations)[0],48,48))
    
    
    for idx,face_coords in enumerate(face_locations):       
        face=img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
        #square crop still missing, there might be some stretching right now
        face_48=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
        faces[idx]=cv2.cvtColor(face_48, cv2.COLOR_BGR2GRAY)
        
        if idx==0:
            face_label0 = tk.Label(frame2)
            face_label0.place(x=5,y=5,width=48,height=48)
            face_img = Image.fromarray(cv2.cvtColor(face_48, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=face_img)
            face_label0.imgtk = imgtk
            face_label0.configure(image=imgtk)
            
            
        if idx==1:
            face_label1 = tk.Label(frame2)
            face_label1.place(x=5,y=5+idx*(48+10),width=48,height=48)
            face_img = Image.fromarray(cv2.cvtColor(face_48, cv2.COLOR_BGR2RGBA))
            imgtk = ImageTk.PhotoImage(image=face_img)
            face_label1.imgtk = imgtk
            face_label1.configure(image=imgtk)
        
    
    model=tf.keras.models.load_model('models/'+model_name.get()+'.model')
#    faces=tf.keras.utils.normalize(faces,axis=1)
    faces=faces/255
    predictions=model.predict(faces.reshape(np.shape(face_locations)[0],48,48,1))
    
    
    emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    
    for idx, face in enumerate(faces):
            plt.imshow(face,cmap='gray')
            plt.show()
            emotion=6
            high_score=0.0
            for i in range(7):
                if predictions[idx,i]>high_score:
                    high_score=predictions[idx,i]
                    emotion=i
                print(emotions[i],':',predictions[idx,i])
            print(emotions[emotion].upper())
            
            if idx==0:
                Text="%s: %.2f" % (emotions[emotion],predictions[idx,emotion])
                emotion_label0 = tk.Label(frame2,text=Text)
                emotion_label0.place(x=5+80,y=5,width=200,height=48)
            
            if idx==1:
                Text="%s: %.2f" % (emotions[emotion],predictions[idx,emotion])
                emotion_label1 = tk.Label(frame2,text=Text)
                emotion_label1.place(x=5+80,y=5+idx*(48+10),width=200,height=48)
    
    
        
    
    
root=tk.Tk()
root.title("Emotion Recognition")
root.minsize(400,400)
root.geometry("800x600")
#root.attributes('-alpha', 0.3)
height=600
width=600


canvas=tk.Canvas(root,height=height,width=width)
canvas.pack()

frame=tk.Frame(root,bg='#80c1ff',bd=5)
frame.place(relx=0.5,relwidth=0.8,relheight=0.5,rely=0.05,anchor='n')

button=tk.Button(frame,text="Start",bg='green',command=start_video_capture)
button.place(relx=0,rely=1,width=100,height=60,anchor='sw')

button=tk.Button(frame,text="Stop",bg='green',command=stop_video_capture)
button.place(relx=1,rely=1,width=100,height=60,anchor='se')

button=tk.Button(frame,text="Capture",bg='green',command=take_photo)
button.place(relx=0.5,rely=1,width=100,height=60,anchor='s')
    


frame2=tk.Frame(root,bg='#ffc180',bd=5)
frame2.place(relx=0.5,relwidth=0.8,relheight=0.35,rely=0.95,anchor='s')

button=tk.Button(frame2,text="Run Model",bg='green',command=Run_model)
button.place(relx=0.5,rely=1,width=100,height=60,anchor='s')

model_name=tk.StringVar()
model_name_entry=tk.Entry(frame2,textvariable=model_name )
model_name.set('64,32,32,3')
model_name_entry.place(relx=0.95,rely=1,anchor='se',height=60,relwidth=0.3)

model_names=tk.Listbox(frame2)
model_names.place(relx=1,rely=0,width=250,relheight=0.6,anchor='ne')

for item in os.listdir('./models'):
    model_names.insert(tk.END, item)

root.mainloop()
if video_capture==True:
    cap.release()
