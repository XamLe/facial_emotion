import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np

class Faces:
	def __init__(self,root,_relx,_relwidth,_rely,_relheight,_anchor):
		self.root=root
		self._relx=_relx
		self._relwidth=_relwidth
		self._relheight=_relheight
		self._rely=_rely
		self._anchor=_anchor
		self.show()
		
	def show(self):
		self.faces_frame=tk.Frame(self.root,bg='#ffc180',bd=5)
		self.faces_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		faces_label=tk.Label(self.faces_frame,text = 'Found Faces')
		faces_label.pack()


	def set_faces(self,faces):
		self.show()
		self.faces=faces
		self.show_faces()

	def set_expression(self,idx,expression,probability):
		Text="%s: %.2f" % (expression,probability)
		emotion_label = tk.Label(self.faces_frame,text=Text)
		emotion_label.place(x=5+80,y=40+idx*(48+10),width=200,height=48)

	def show_faces(self):
		for idx,face in enumerate(self.faces):
			face_label = tk.Label(self.faces_frame)
			face_label.place(x=5,y=40+idx*(48+10),width=48,height=48)
			face_img = Image.fromarray(np.uint8(face))
			imgtk = ImageTk.PhotoImage(image=face_img)
			face_label.imgtk = imgtk
			face_label.configure(image=imgtk)
