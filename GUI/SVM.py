import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pickle
from sklearn import svm
import os

class SVM:

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

	def show(self):
		self.frame=tk.Frame(self.root,bg='#ffc180',bd=5)
		self.frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		label=tk.Label(self.frame,text = 'SVM')
		label.pack()

		button=tk.Button(self.frame,text="Run Model",command=self.pressed_Run_model)
		button.place(relx=1,rely=1,width=100,height=60,anchor='se')

		self.model_names=tk.Listbox(self.frame)
		self.model_names.place(relx=0,rely=0.55,width=250,relheight=0.8,anchor='w')
		for item in os.listdir('./svm_models'):
		    self.model_names.insert(tk.END, item)
		# self.model_names.bind("<<ListboxSelect>>",self.load_selected_model)
		self.model_names.select_set(0)


	def load_selected_model(self,value):
		self.model_name=self.model_names.get(self.model_names.curselection())

		with open('svm_models/'+self.model_name, 'rb') as fid:
			self.model = pickle.load(fid);

	def pressed_Run_model(self):
		self.load_selected_model(0)
		self.Run_model()

	def Run_model(self):

		faces=self.input.get_faces_grayscale()

		
		# predictions=[self.model.predict(faces.reshape(np.shape(faces)[0],48,48,1))]

		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
		predictions = []

		for idx, face in enumerate(faces):
			emotion = self.model.predict(face.reshape(1, 2304) / 255)
			self.faces.set_expression(idx,expression=emotions[emotion[0]],probability=1)



		# for idx, face in enumerate(faces):
		# 	emotion=6
		# 	highscore=0.0
		# 	for i in range(7):
		# 		if predictions[idx,i]>highscore:
		# 		        highscore=predictions[idx,i]
		# 		        emotion=i


		 	