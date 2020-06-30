import tkinter as tk
from tkinter.ttk import Progressbar
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import face_recognition
import cv2
import _thread


class Train_CNN:
	def __init__(self,root,_relx,_relwidth,_rely,_relheight,_anchor):
		self.root=root
		self._relx=_relx
		self._relwidth=_relwidth
		self._relheight=_relheight
		self._rely=_rely
		self._anchor=_anchor
		self.show()
		
	def show(self):
		self.frame=tk.Frame(self.root,bg='#80e0a1',bd=5)
		self.frame.place(relx=self._relx,relwidth=self._relwidth,rely=self._rely,relheight=self._relheight,anchor=self._anchor)

		label=tk.Label(self.frame,text = 'Train CNN')
		label.pack()

		load_button=tk.Button(self.frame,text="Train CNN",command=self.train)
		load_button.place(relx=1,rely=1,width=90,height=35,anchor='se')

		self.model_name_entry=tk.Entry(self.frame)
		self.model_name_entry.place(anchor='nw',x=55,y=30,width=130)
		self.model_name_entry.insert(tk.END,'new Model name')

		label=tk.Label(self.frame,text='Name:',bg=self.frame["background"])
		label.place(anchor='nw',x=0,y=34)

		label=tk.Label(self.frame,text='Epochs:',bg=self.frame["background"])
		label.place(anchor='nw',x=0,y=69)

		self.epochs_entry=tk.Entry(self.frame)
		self.epochs_entry.place(anchor='nw',x=55,y=65,width=130)
		self.epochs_entry.insert(tk.END,'1')



		self.progressbar=Progressbar(self.frame,maximum = 100,mode='indeterminate')
		self.progressbar.place(anchor='sw',x=0,rely=1)

		emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
		self.checklist = ChecklistBox(self.frame, emotions, bd=1, relief="sunken", background="white")
		self.checklist.place(anchor='ne',relx=1,y=0.38)


	def train(self):
		_thread.start_new_thread(self.train_network,())


	def train_network(self):
		name=self.model_name_entry.get()
		epochs=int(self.epochs_entry.get())

		if not (self.checklist.get_deactivated_expressions() == []):
			trainingdata = open("/Users/jonaseckhoff/facial_emotion_local/28709train.txt","r")

			x_train_filtered=[]
			y_train_filtered=[]


			deactivated_expressions=self.checklist.get_deactivated_expressions()
			for i in range(28709):
			    data=trainingdata.readline() #read one image
			    if i%500==0:
			        self.progressbar['value']=int(i*100/28709)
			    if int(data[0]) in deactivated_expressions:
			        continue
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
			        x_train_filtered.append(face_48)
			        y_train_filtered.append(int(data[0]))
			    
			trainingdata.close()

			x_train=np.empty((len(x_train_filtered),48,48,1))
			y_train=np.empty(len(y_train_filtered))


			for i in range(len(x_train_filtered)):
			    x_train[i]=x_train_filtered[i].reshape(48,48,1)
			    y_train[i]=y_train_filtered[i]


		else:
			self.progressbar.start()
			x_train=np.load('image_data.npy')
			y_train=np.load('emotion_data.npy')


		test_size=1000
		x_test=x_train[-test_size:-1]
		x_train=x_train[0:-test_size]
		y_test=y_train[-test_size:-1]
		y_train=y_train[0:-test_size]

		#x_train=tf.keras.utils.normalize(x_train,axis=1)
		#x_test=tf.keras.utils.normalize(x_test,axis=1)
		x_train=x_train/255
		x_test=x_test/255

		#convolutional neural network
		model = tf.keras.models.Sequential()
		model.add(Conv2D(64, (3,3) ,input_shape = x_train.shape[1:]))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(64, (3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))


		model.add(Flatten())
		model.add(Dense(32))
		model.add(Activation('relu'))

		model.add(Dense(7))
		model.add(Activation(tf.nn.softmax))


		model.compile(optimizer='adam',
		              loss='sparse_categorical_crossentropy',
		              metrics=['accuracy'])



		model.fit(x_train,y_train, epochs=epochs)

		val_loss,val_acc=model.evaluate(x_test,y_test)
		print(val_loss,val_acc)

		model.save('models/'+name+'.model')
		self.progressbar.stop()



class ChecklistBox(tk.Frame):
    def __init__(self, parent, choices, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)

        self.vars = []
        bg = self.cget("background")
        for choice in choices:
            var = tk.StringVar(value=choice)
            self.vars.append(var)
            cb = tk.Checkbutton(self, var=var, text=choice,
                                onvalue=choice, offvalue="",
                                anchor="w", width=10, background=bg,
                                relief="flat", highlightthickness=0
            )
            cb.pack(side="top", fill="x", anchor="w")


    def getCheckedItems(self):
        values = []
        for var in self.vars:
            value =  var.get()
            if value:
                values.append(value)
        return values

    def get_deactivated_expressions(self):
    	idxs=[]
    	for idx,var in enumerate(self.vars):
    	    value =  var.get()
    	    if not value:
    	        idxs.append(idx)
    	return idxs