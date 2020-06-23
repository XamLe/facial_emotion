import tkinter as tk
import cv2
from mss import mss
from PIL import Image, ImageTk
import numpy as np

class Screengrab:

	def __init__(self,root,input,_relx,_relwidth,_rely,_relheight,_anchor):
	    self.root=root
	    self._relx=_relx
	    self._relwidth=_relwidth
	    self._relheight=_relheight
	    self._rely=_rely
	    self._anchor=_anchor
	    self.show()
	    self.input=input
	    self.sct=mss()


	def take_screenshot(self):
	    

	    framecap=self.sct.grab({'top': int(self.top_s.get()), 'left': int(self.left_s.get()), 'width': int(self.width_s.get()), 'height': int(self.height_s.get())})
	    img = Image.frombytes("RGB", framecap.size, framecap.bgra, "raw", "BGRX")
	    framecap=cv2.resize(np.uint8(img),(0,0),fx=1,fy=1,interpolation = cv2.INTER_AREA)
	    self.input.set_input(framecap)


	def start_capture(self):
		
		self.screen_capture = tk.Label(self.screen_grab_frame,bg='#80c1ff')
		self.screen_capture.place(relx=0.55,rely=0.1,relwidth=0.8,relheight=0.32,anchor='n')
		self.Screen_Capture=True

		def show_frame():
			if self.Screen_Capture==True:
			    framecap=self.sct.grab({'top': int(self.top_s.get()), 'left': int(self.left_s.get()), 'width': int(self.width_s.get()), 'height': int(self.height_s.get())})
			    framecap = Image.frombytes("RGB", framecap.size, framecap.bgra, "raw", "BGRX")
			    cv2image=cv2.resize(np.uint8(framecap),(0,0),fx=0.25,fy=0.25,interpolation = cv2.INTER_AREA)
			    img = Image.fromarray(cv2image)
			    imgtk = ImageTk.PhotoImage(image=img)
			    self.screen_capture.imgtk = imgtk
			    self.screen_capture.configure(image=imgtk)
			    self.screen_capture.after(10, show_frame)
		show_frame()

	def stop_capture(self): 
		self.Screen_Capture=False
		self.screen_capture.destroy()

	def x_set(self,value):
		self.left_s.delete(0,tk.END)
		self.left_s.insert(tk.END,value)

	def y_set(self,value):
		self.top_s.delete(0,tk.END)
		self.top_s.insert(tk.END,value)


	def show(self):
	    self.screen_grab_frame=tk.Frame(self.root,bg='#80c1ff',bd=5)
	    self.screen_grab_frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

	    input_screen_capture_label=tk.Label(self.screen_grab_frame,text = 'Screen Capture')
	    input_screen_capture_label.pack()
	    
	    button=tk.Button(self.screen_grab_frame,text="Capture",command=self.take_screenshot)
	    button.place(relx=0.5,rely=1,width=100,height=60,anchor='s')

	    button=tk.Button(self.screen_grab_frame,text="Start",bg='green',command=self.start_capture)
	    button.place(relx=0,rely=1,width=100,height=60,anchor='sw')	    

	    button=tk.Button(self.screen_grab_frame,text="Stop",bg='green',command=self.stop_capture)
	    button.place(relx=1,rely=1,width=100,height=60,anchor='se')

	    self.top_s=tk.Entry(self.screen_grab_frame)
	    self.top_s.place(relwidth=0.4,relx=0.05,rely=0.75,anchor='w')
	    self.left_s=tk.Entry(self.screen_grab_frame)
	    self.left_s.place(relwidth=0.4,relx=0.05,rely=0.65,anchor='w')
	    self.width_s=tk.Entry(self.screen_grab_frame)
	    self.width_s.place(relwidth=0.4,relx=0.95,rely=0.65,anchor='e')
	    self.height_s=tk.Entry(self.screen_grab_frame)
	    self.height_s.place(relwidth=0.4,relx=0.95,rely=0.75,anchor='e')

	    self.slider_x=tk.Scale(self.screen_grab_frame,from_=10,to=1000,bg='#80c1ff',command=self.y_set)
	    self.slider_x.set(300)

	    self.slider_x.place(anchor='ne',y=30,x=50)

	    self.slider_y=tk.Scale(self.screen_grab_frame,from_=10,to=1000,bg='#80c1ff',command=self.x_set,orient=tk.HORIZONTAL)
	    self.slider_y.set(170)
	    self.slider_y.place(anchor='s',rely=0.55,relx=0.5)


	    
	    self.top_s.insert(tk.END,'170')
	    self.left_s.insert(tk.END,'300')
	    self.width_s.insert(tk.END,'600')
	    self.height_s.insert(tk.END,'310')