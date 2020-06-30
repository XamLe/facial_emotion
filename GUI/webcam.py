import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np

class Webcam:

	def __init__(self,root,input,_relx,_relwidth,_rely,_relheight,_anchor):
	        self.video_capture=False
	        self.root=root
	        self._relx=_relx
	        self._relwidth=_relwidth
	        self._relheight=_relheight
	        self._rely=_rely
	        self._anchor=_anchor
	        self.show()
	        self.input=input
    
	def start_video_capture(self):
		if self.video_capture==False:
			self.video_capture=True


			self.cap = cv2.VideoCapture(0)
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
			self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

			self.camera = tk.Label(self.webcam_frame,bg='#80c1ff')
			self.camera.place(relx=0.5,rely=0.15,relwidth=0.8,relheight=0.6,anchor='n')

			def show_frame():
				if self.video_capture==True:
				    _, webcam_image = self.cap.read()
				    webcam_image = cv2.flip(webcam_image, 1)
				    cv2image = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGBA)
				    cv2image=cv2.resize(cv2image,(0,0),fx=0.2,fy=0.2,interpolation = cv2.INTER_AREA)
				    img = Image.fromarray(cv2image)
				    imgtk = ImageTk.PhotoImage(image=img)
				    self.camera.imgtk = imgtk
				    self.camera.configure(image=imgtk)
				    self.camera.after(10, show_frame)

			show_frame()

	def stop_video_capture(self): 
	    if self.video_capture==True:
	        self.video_capture=False
	    self.camera.destroy()
	    self.cap.release()


	def take_photo(self):
		if self.video_capture==True:
			_, framecap = self.cap.read()
			# framecap = cv2.flip(framecap, 1)
			cv2image = cv2.cvtColor(framecap, cv2.COLOR_BGR2RGB)
			self.input.set_input(cv2image)
	        

	def show(self):

		self.webcam_frame=tk.Frame(self.root,bg='#80c1ff',bd=5)
		self.webcam_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		input_method_label=tk.Label(self.webcam_frame,text = 'Webcam Input')
		input_method_label.pack()

		button=tk.Button(self.webcam_frame,text="Start",bg='green',command=self.start_video_capture)
		button.place(relx=0,rely=1,width=90,height=35,anchor='sw')

		button=tk.Button(self.webcam_frame,text="Stop",bg='green',command=self.stop_video_capture)
		button.place(relx=1,rely=1,width=90,height=35,anchor='se')

		button=tk.Button(self.webcam_frame,text="Capture",bg='green',command=self.take_photo)
		button.place(relx=0.5,rely=1,width=90,height=35,anchor='s')

