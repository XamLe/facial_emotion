import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition


class Input_Image:
	def __init__(self,root,faces,_relx,_relwidth,_rely,_relheight,_anchor):
		self.root=root
		self._relx=_relx
		self._relwidth=_relwidth
		self._relheight=_relheight
		self._rely=_rely
		self._anchor=_anchor
		self.show()
		self.faces=faces
		
	def show(self):
		self.input_Image_frame=tk.Frame(self.root,bg='#ffc180',bd=5)
		self.input_Image_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		self.input_image=tk.Label(self.input_Image_frame,bg='#80c1ff')
		self.input_image.place(relx=0.5,rely=0.45,relwidth=0.8,relheight=0.6,anchor='center')


		input_Image_label=tk.Label(self.input_Image_frame,text = 'Input Image')
		input_Image_label.pack()

		self.input_path=tk.Entry(self.input_Image_frame)
		self.input_path.place(relx=0,rely=0.95,width=200,height=30,anchor='sw')

		load_button=tk.Button(self.input_Image_frame,text="Load Image",command=self.load_image)
		load_button.place(relx=1,rely=1,width=150,height=60,anchor='se')

	def load_image(self):
		img=cv2.imread(self.input_path.get())
		self.set_input(img)


	def set_input(self,img):
		self.image=img
		self.show_input()

		face_locations = face_recognition.face_locations(img)
		faces=np.empty((np.shape(face_locations)[0],48,48,3))


		for idx,face_coords in enumerate(face_locations):       
		    face=img[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
		    #square crop still missing, there might be some stretching right now
		    faces[idx]=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)

		self.faces.set_faces(faces)


	def show_input(self):
		img=cv2.resize(np.array(self.image),(0,0),fx=0.3,fy=0.3,interpolation = cv2.INTER_AREA)
		img = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image=img)
		self.input_image.imgtk = imgtk
		self.input_image.configure(image=imgtk)

	def get_input(self):
		return self.image

	def get_faces(self):
		return self.faces.faces

	def get_faces_grayscale(self):
		faces=np.empty((self.faces.faces.shape[0],48,48))
		for idx, face in enumerate(self.faces.faces):
			faces[idx]=cv2.cvtColor(np.uint8(face), cv2.COLOR_RGB2GRAY)
		return faces
