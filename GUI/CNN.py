import tensorflow as tf
import numpy as np
import tkinter as tk
import face_recognition
import cv2
from PIL import Image, ImageTk
import os

class CNN:

	def __init__(self,root,input,faces,_relx,_relwidth,_rely,_relheight,_anchor):
	        self.root=root
	        self._relx=_relx
	        self._relwidth=_relwidth
	        self._relheight=_relheight
	        self._rely=_rely
	        self._anchor=_anchor
	        self.show()
	        self.input=input
	        self.faces=faces
	        self.load_selected_model(0)
	        self.model=tf.keras.models.load_model('models/hsn.model')

	def show(self):
		self.CNN_frame=tk.Frame(self.root,bg='#80e0a1',bd=5)
		self.CNN_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		CNN_label=tk.Label(self.CNN_frame,text = 'CNN')
		CNN_label.pack()

		button=tk.Button(self.CNN_frame,text="Run Model",command=self.pressed_Run_model)
		button.place(relx=1,rely=1,width=90,height=35,anchor='se')

		self.model_names=tk.Listbox(self.CNN_frame)
		self.model_names.place(relx=0,rely=1,relwidth=0.65,relheight=0.87,anchor='sw')
		for item in os.listdir('./models'):
			if item[0]!='.':
				self.model_names.insert(tk.END, item[0:-6])
		# self.model_names.bind("<<ListboxSelect>>",self.load_selected_model)
		self.model_names.select_set(0)


	def load_selected_model(self,event):
		self.model_name=self.model_names.get(self.model_names.curselection())
		self.model=tf.keras.models.load_model('models/'+self.model_name+'.model')

	def pressed_Run_model(self):
		try:
			self.load_selected_model(0)
		except:
			pass
		self.Run_model()


	def Run_model(self):
		try:
			faces=self.input.get_faces_grayscale()
			if faces.shape[0] == 0:
				return
		except:
			try:
				faces=self.input.get_faces()
			except:
				return

		faces=faces/255

		predictions=self.model.predict(faces.reshape(np.shape(faces)[0],48,48,1))

		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

		for idx, face in enumerate(faces):
			emotion=6
			high_score=0.0
			for i in range(7):
				if predictions[idx,i]>high_score:
				        high_score=predictions[idx,i]
				        emotion=i


			self.faces.set_expression(idx,expression=emotions[emotion],probability=predictions[idx,emotion],position=0)

	def single_eval(self,face):
		face=face/255
		predictions=self.model.predict(face.reshape(1,48,48,1))
		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
		emotion=6
		high_score=0.0
		for i in range(7):
			if predictions[0,i]>high_score:
			        high_score=predictions[0,i]
			        emotion=i
		self.faces.set_expression(0,expression=emotions[emotion],probability=predictions[0,emotion],position=0)
		return emotion
