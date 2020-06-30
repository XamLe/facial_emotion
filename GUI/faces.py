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
		self.labels=[]

		
	def show(self):
		self.faces_frame=tk.Frame(self.root,bg='#ffc180',bd=5)
		self.faces_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		faces_label=tk.Label(self.faces_frame,text = 'Found Faces')
		faces_label.pack()

		label=tk.Label(self.faces_frame,text='CNN')
		label.place(x=110,y=30,anchor='n',width=100)

		label=tk.Label(self.faces_frame,text='AdaBoost')
		label.place(x=110+105,y=30,anchor='n',width=100)

	def set_faces(self,faces):
		self.faces=faces

		for label in self.labels:
			label.destroy()
		self.labels=[]
		self.show_faces()

	def set_expression(self,idx,expression,probability,position):
		if probability==-1:
			Text=expression
		else:
			Text="%s: %.2f" % (expression,probability)
		emotion_label = tk.Label(self.faces_frame,text=Text)
		self.labels.append(emotion_label)
		emotion_label.place(x=60+position*105,y=60+idx*(48+10),width=100,height=48,anchor='nw')

#:)
# :O
# :|
# :(
# >:o

	def show_faces(self):
		for idx,face in enumerate(self.faces):
			face_label = tk.Label(self.faces_frame)
			self.labels.append(face_label)
			face_label.bind("<Button-1>", self.show_face)
			face_label.place(x=5,y=60+idx*(48+10),width=48,height=48)
			face_img = Image.fromarray(np.uint8(face))
			imgtk = ImageTk.PhotoImage(image=face_img)
			face_label.imgtk = imgtk
			face_label.configure(image=imgtk)

	def show_face(self,event):
		label=event.widget
		face=tk.Toplevel(event.widget.master.master)
		face.title("Face")
		face.minsize(480,480)
		face.geometry(str(480)+'x'+str(480)+'+'+str(self.root.winfo_x()+400)+'+'+str(self.root.winfo_y()+136))
		face_label=tk.Label(face)
		face_label.pack()
		img=label.imgtk._PhotoImage__photo.zoom(10, 10)
		face_label.imgtk=img
		face_label.configure(image=img)
