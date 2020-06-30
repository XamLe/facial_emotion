import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class Algorithm:

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
		self.frame=tk.Frame(self.root,bg='#80e0a1',bd=5)
		self.frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		label=tk.Label(self.frame,text = 'Algorithm')
		label.pack()

		button=tk.Button(self.frame,text="Run Model",command=self.Run_model)
		button.place(relx=1,rely=1,width=90,height=35,anchor='se')



	def Run_model(self):

		faces=self.input.get_faces_grayscale()

		
		# predictions=[self.model.predict(faces.reshape(np.shape(faces)[0],48,48,1))]

		# emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

		# for idx, face in enumerate(faces):
		# 	emotion=6
		# 	highscore=0.0
		# 	for i in range(7):
		# 		if predictions[idx,i]>highscore:
		# 		        highscore=predictions[idx,i]
		# 		        emotion=i


		# 	self.faces.set_expression(idx,expression=emotions[emotion],probability=predictions[idx,emotion])

