import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import pickle
import os

class Adaboost:

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

			pkl_filename="models_adaboost/adaboost_model_n_100_r_1.pkl"
			with open(pkl_filename, 'rb') as file:
				self.pickle_model = pickle.load(file)

	def show(self):
		self.frame=tk.Frame(self.root,bg='#80e0a1',bd=5)
		self.frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		label=tk.Label(self.frame,text = 'AdaBoost')
		label.pack()

		button=tk.Button(self.frame,text="Run Model",command=self.pressed_Run_model)
		button.place(relx=1,rely=1,width=100,height=35,anchor='se')

		self.model_names=tk.Listbox(self.frame)
		self.model_names.place(relx=0,rely=1,relwidth=0.65,relheight=0.87,anchor='sw')
		for item in os.listdir('./models_adaboost'):
			if item[0]!='.':
				self.model_names.insert(tk.END, item[0:-4])
		# self.model_names.bind("<<ListboxSelect>>",self.load_selected_model)
		self.model_names.select_set(0)


	def load_selected_model(self,event):
		self.model_name=self.model_names.get(self.model_names.curselection())

		pkl_filename="models_adaboost/"+self.model_name+".pkl"
		with open(pkl_filename, 'rb') as file:
			self.pickle_model = pickle.load(file)

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

		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
		

		face=faces.reshape(faces.shape[0],48*48)
		face=np.uint8(face)

		emotions_ids=self.pickle_model.predict(face)
		
		for idx, emotion_id in enumerate(emotions_ids):
			self.faces.set_expression(idx,expression=emotions[emotion_id],probability=-1,position=1)

	def single_eval(self,face):
			emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

			prediction=self.pickle_model.predict(np.uint8(face.reshape(1,48*48)))
			self.faces.set_expression(0,expression=emotions[prediction[0]],probability=-1,position=1)

			return prediction[0]
