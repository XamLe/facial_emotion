import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition

class Trainingsdata:
	def __init__(self,root,input,cnn,adaboost,_relx,_relwidth,_rely,_relheight,_anchor):
		self.root=root
		self._relx=_relx
		self._relwidth=_relwidth
		self._relheight=_relheight
		self._rely=_rely
		self._anchor=_anchor
		self.show()
		self.input=input
		self.cnn=cnn
		self.adaboost=adaboost

		
	def show(self):
		self.faces_frame=tk.Frame(self.root,bg='#80c1ff',bd=5)
		self.faces_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		faces_label=tk.Label(self.faces_frame,text = 'Image data')
		faces_label.pack()

		button=tk.Button(self.faces_frame,text="Show data",command=self.show_face_window)
		button.place(relx=1,rely=1,width=100,height=35,anchor='se')

		button=tk.Button(self.faces_frame,text="Show filtered data",command=self.show_face_window_filtered)
		button.place(relx=1,rely=0.8,width=140,height=35,anchor='se')

		label=tk.Label(self.faces_frame,bg='#80c1ff',text='Data:')
		label.place(x=0,y=43, width=50,anchor='nw')

		self.dataname=tk.Entry(self.faces_frame)
		self.dataname.place(y=40,x=55,width=200,anchor='nw')

		self.range_min=tk.Entry(self.faces_frame)
		self.range_min.place(relx=0,rely=1,width=50,height=35,anchor='sw')

		self.range_max=tk.Entry(self.faces_frame)
		self.range_max.place(relx=0.2,rely=1,width=50,height=35,anchor='sw')

		self.range_filtered_min=tk.Entry(self.faces_frame)
		self.range_filtered_min.place(relx=0,rely=0.8,width=50,height=35,anchor='sw')

		self.range_filtered_max=tk.Entry(self.faces_frame)
		self.range_filtered_max.place(relx=0.2,rely=0.8,width=50,height=35,anchor='sw')

		self.range_min.insert(tk.END,'0')
		self.range_max.insert(tk.END,'100')
		self.range_filtered_min.insert(tk.END,'0')
		self.range_filtered_max.insert(tk.END,'100')
		self.dataname.insert(tk.END,'28709train.txt')


		label=tk.Label(self.faces_frame,text = 'range',bg='#80c1ff')
		label.place(relx=0.17,anchor='s',rely=0.63)

	def show_face_window(self):

		self.face_window=tk.Toplevel(self.root)
		self.face_window.title("Image data")
		self.face_window.minsize(600,640)
		height=640
		width=600
		self.face_window.geometry(str(width)+'x'+str(height)+'+'+str(self.root.winfo_x()+340)+'+'+str(self.root.winfo_y()+60))

		self.count=0

		self.show_faces()

	def show_face_window_filtered(self):

		self.face_window_filtered=tk.Toplevel(self.root)
		self.face_window_filtered.title("Image data filtered")
		self.face_window_filtered.minsize(600,640)
		height=780
		width=600
		self.face_window_filtered.geometry(str(width)+'x'+str(height)+'+'+str(self.root.winfo_x()+340)+'+'+str(self.root.winfo_y()+60))


		self.count_filtered=0
		self.show_faces_filtered()

	def more_faces(self):
		self.count+=1
		self.face_frame.destroy()
		self.show_faces()

	def more_faces_filtered(self):
		self.count_filtered+=1
		self.face_frame_filtered.destroy()
		self.show_faces_filtered()

	def show_faces(self):

		self.face_frame=tk.Frame(self.face_window,bg='#80e0a1',bd=5)
		self.face_frame.place(relx=0.5,rely=0.5,anchor='center',width=580,height=620)

		button=tk.Button(self.face_frame,text="More",command=self.more_faces)
		button.place(relx=1,rely=1,width=70,height=35,anchor='se')

		button=tk.Button(self.face_frame,text="Evaluate all",command=self.evaluate_all_unfiltered)
		button.place(relx=1,rely=0.92,width=80,height=35,anchor='se')
		trainingdata = open("/Users/jonaseckhoff/facial_emotion_local/"+self.dataname.get(),"r")			
		self.faces_unfiltered=[]
		for i in range(7):
			self.faces_unfiltered.append([])
		for i in range(self.count*1000+int(self.range_min.get())):
			trainingdata.readline()
		for i in range(int(self.range_max.get())-int(self.range_min.get())):
			    data=trainingdata.readline() #read one image
			    emotion=int(data[0])
			    # if int(data[0]) in deactivated_expressions:
			    #     continue
			    image_string=data[3:-2] #cut off irrelevant part of the string
			    image=np.array([int(k) for k in image_string.split(' ')])
			    image=image.reshape(48,48)
			    image=np.uint8(image)
			    self.faces_unfiltered[emotion].append((image,i))
			    
		trainingdata.close()
		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

		for y in range(7):
			label=tk.Label(self.face_frame,text = emotions[y],bg='#80e0a1')
			label.place(x=5,y=40+y*(48+10+20),height=48)
			for x in range(7):


				try:
					self.faces_unfiltered[y][x][0]
					face_label = tk.Label(self.face_frame)
					face_label.place(x=88+x*(48+10),y=40+y*(48+10+20),width=48,height=48)
					face_label.bind("<Button-1>", self.show_face)
					face_label.bind("<ButtonPress-2>", lambda event, x_=x,y_=y:self.evaluate_face_unfiltered(x_,y_))
					face_img = Image.fromarray(np.uint8(self.faces_unfiltered[y][x][0]))
					imgtk = ImageTk.PhotoImage(image=face_img)
					face_label.imgtk = imgtk
					face_label.configure(image=imgtk)
					label=tk.Label(self.face_frame,text = self.faces_unfiltered[y][x][1]+self.count*1000+int(self.range_min.get()),bg='#80e0a1')
					label.place(x=88+x*(48+10),y=35+58+y*(48+10+20),width=48,height=20)
				except:
					continue

	def show_faces_filtered(self):

		self.face_frame_filtered=tk.Frame(self.face_window_filtered,bg='#80e0a1',bd=5)
		self.face_frame_filtered.place(relx=0.5,rely=0.5,anchor='center',width=580,height=760)

		button=tk.Button(self.face_frame_filtered,text="More",command=self.more_faces_filtered)
		button.place(relx=1,rely=1,width=80,height=35,anchor='se')

		button=tk.Button(self.face_frame_filtered,text="Evaluate all",command=self.evaluate_all)
		button.place(relx=1,rely=0.92,width=80,height=35,anchor='se')

		trainingdata = open("/Users/jonaseckhoff/facial_emotion_local/28709train.txt","r")			
		self.faces=[]
		self.removed=[]
		for i in range(7):
			self.faces.append([])
		for i in range(self.count_filtered*1000+int(self.range_filtered_min.get())):
			trainingdata.readline()
		for i in range(int(self.range_filtered_max.get())-int(self.range_filtered_min.get())):
			    data=trainingdata.readline() #read one image
			    emotion=int(data[0])
			    # if int(data[0]) in deactivated_expressions:
			    #     continue
			    image_string=data[3:-2] #cut off irrelevant part of the string
			    image=np.array([int(k) for k in image_string.split(' ')])
			    image=image.reshape(48,48)
			    image=np.uint8(image)
			    face_locations=face_recognition.face_locations(image)
			    if np.shape(face_locations)[0]==1:
			        face_coords=face_locations[0]
			        face=image[face_coords[0]:face_coords[2],face_coords[3]:face_coords[1]]
			        #square crop still missing, there might be some stretching right now
			        face_48=cv2.resize(face,(48,48),interpolation = cv2.INTER_AREA)
			        self.faces[emotion].append((face_48,i))
			    else:
			    	self.removed.append((image,i))

			    
		trainingdata.close()
		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

		for y in range(7):
			label=tk.Label(self.face_frame_filtered,text = emotions[y],bg='#80e0a1')
			label.place(x=5,y=40+y*(48+10+20),height=48)
			for x in range(7):


				try:
					self.faces[y][x][0]
					face_label = tk.Label(self.face_frame_filtered)
					face_label.place(x=88+x*(48+10),y=40+y*(48+10+20),width=48,height=48)
					face_label.bind("<Button-1>", self.show_face)
					face_label.bind("<ButtonPress-2>", lambda event, x_=x,y_=y:self.evaluate_face(x_,y_))
					face_img = Image.fromarray(np.uint8(self.faces[y][x][0]))
					imgtk = ImageTk.PhotoImage(image=face_img)
					face_label.imgtk = imgtk
					face_label.configure(image=imgtk)
					label=tk.Label(self.face_frame_filtered,text = self.faces[y][x][1]+self.count_filtered*1000+int(self.range_filtered_min.get()),bg='#80e0a1')
					label.place(x=88+x*(48+10),y=35+58+y*(48+10+20),width=48,height=20)
				except:
					continue

		for y in range(7,9):		
			for x in range(7):
				label=tk.Label(self.face_frame_filtered,text = 'removed',bg='#80e0a1')
				label.place(x=5,y=40+y*(48+10+20),height=48)
				try:

					self.removed[x+(y-7)*7][0]
					face_label = tk.Label(self.face_frame_filtered)
					face_label.place(x=88+x*(48+10),y=40+y*(48+10+20),width=48,height=48)
					face_label.bind("<Button-1>", self.show_face)
					face_label.bind("<ButtonPress-2>", lambda event, x_=x,y_=y:self.evaluate_face(x_,y_))
					face_img = Image.fromarray(np.uint8(self.removed[x+(y-7)*7][0]))
					imgtk = ImageTk.PhotoImage(image=face_img)
					face_label.imgtk = imgtk
					face_label.configure(image=imgtk)
					label=tk.Label(self.face_frame_filtered,text = self.removed[x+(y-7)*7][1]+self.count_filtered*1000+int(self.range_filtered_min.get()),bg='#80e0a1')
					label.place(x=88+x*(48+10),y=35+58+y*(48+10+20),width=48,height=20)
				except:
					continue


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

	def evaluate_face(self,x,y):
		self.input.set_input_grayscale(self.faces[y][x][0])
		if self.cnn.single_eval(self.faces[y][x][0])==y:
			label=tk.Frame(self.face_frame_filtered,bg='#00aa00')
			label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')
		else:
			label=tk.Frame(self.face_frame_filtered,bg='#aa0000')
			label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')

		if self.adaboost.single_eval(self.faces[y][x][0])==y:
			label=tk.Frame(self.face_frame_filtered,bg='#00aa00')
			label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
		else:
			label=tk.Frame(self.face_frame_filtered,bg='#aa0000')
			label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')

	def evaluate_face_unfiltered(self,x,y):
		self.input.set_input_grayscale(self.faces_unfiltered[y][x][0])
		if self.cnn.single_eval(self.faces_unfiltered[y][x][0])==y:
			label=tk.Frame(self.face_frame,bg='#00aa00')
			label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')
		else:
			label=tk.Frame(self.face_frame,bg='#aa0000')
			label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')

		if self.adaboost.single_eval(self.faces_unfiltered[y][x][0])==y:
			label=tk.Frame(self.face_frame,bg='#00aa00')
			label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
		else:
			label=tk.Frame(self.face_frame,bg='#aa0000')
			label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')

	def evaluate_all(self):
		for y in range(7):
			for x in range(7):
				try:
					if self.cnn.single_eval(self.faces[y][x][0])==y:
						label=tk.Frame(self.face_frame_filtered,bg='#00aa00')
						label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')
					else:
						label=tk.Frame(self.face_frame_filtered,bg='#aa0000')
						label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')

					if self.adaboost.single_eval(self.faces[y][x][0])==y:
						label=tk.Frame(self.face_frame_filtered,bg='#00aa00')
						label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
					else:
						label=tk.Frame(self.face_frame_filtered,bg='#aa0000')
						label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
				except:
					pass

	def evaluate_all_unfiltered(self):
		for y in range(7):
			for x in range(7):
				try:
					if self.cnn.single_eval(self.faces_unfiltered[y][x][0])==y:
						label=tk.Frame(self.face_frame,bg='#00aa00')
						label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')
					else:
						label=tk.Frame(self.face_frame,bg='#aa0000')
						label.place(x=88+x*(48+10)-5,y=40+y*(48+10+20)-5,width=15,height=15,anchor='nw')

					if self.adaboost.single_eval(self.faces_unfiltered[y][x][0])==y:
						label=tk.Frame(self.face_frame,bg='#00aa00')
						label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
					else:
						label=tk.Frame(self.face_frame,bg='#aa0000')
						label.place(x=88+x*(48+10)+24,y=40+y*(48+10+20)-5,width=15,height=15,anchor='n')
				except:
					pass


