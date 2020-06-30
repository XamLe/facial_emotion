import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class Video:

	def __init__(self,root,input,screengrab,cnn,adaboost,cam,_relx,_relwidth,_rely,_relheight,_anchor):
	        self.root=root
	        self._relx=_relx
	        self._relwidth=_relwidth
	        self._relheight=_relheight
	        self._rely=_rely
	        self._anchor=_anchor
	        self.show()
	        self.input=input
	        self.screengrab=screengrab
	        self.cnn=cnn
	        self.cam=cam
	        self.capture=False
	        self.capture_webcam=False
	        self.adaboost=adaboost


	def show(self):
		self.frame=tk.Frame(self.root,bg='#80c1ff',bd=5)
		self.frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		label=tk.Label(self.frame,text = 'Video')
		label.pack()

		button=tk.Button(self.frame,text="Start",command=self.start)
		button.place(relx=0,rely=1,width=90,height=35,anchor='sw')

		button=tk.Button(self.frame,text="Stop",command=self.stop)
		button.place(relx=1,rely=1,width=90,height=35,anchor='se')

		self.frametime=tk.Entry(self.frame)
		self.frametime.place(anchor='s',relx=0.5,rely=1,width=50)
		self.frametime.insert(tk.END,'1000')

		button=tk.Button(self.frame,text="Start",command=self.start_webcam)
		button.place(relx=0,rely=0.8,width=90,height=35,anchor='sw')

		button=tk.Button(self.frame,text="Stop",command=self.stop_webcam)
		button.place(relx=1,rely=0.8,width=90,height=35,anchor='se')

		self.frametime_webcam=tk.Entry(self.frame)
		self.frametime_webcam.place(anchor='s',relx=0.5,rely=0.8,width=50)
		self.frametime_webcam.insert(tk.END,'1000')

	def stop(self):
		self.capture=False

	def start(self):
		if self.capture_webcam==True:
			self.capture_webcam=False
		self.capture=True
		def take_frame():
			if self.capture==True:
				self.screengrab.take_screenshot()
				self.cnn.Run_model()
				self.adaboost.Run_model()
				self.frame.after(int(self.frametime.get()),take_frame)
		take_frame()
		

	def stop_webcam(self):
		self.capture_webcam=False

	def start_webcam(self):
		if self.capture==True:
			self.capture=False
		if self.cam.video_capture==False:
			self.cam.start_video_capture()
		self.capture_webcam=True
		def take_frame():
			if self.capture_webcam==True:
				self.cam.take_photo()
				self.cnn.Run_model()
				self.adaboost.Run_model()
				self.frame.after(int(self.frametime.get()),take_frame)
		take_frame()