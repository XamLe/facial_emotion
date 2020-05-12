import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import face_recognition
import cv2

trainingdata = open("28709train.txt","r")

x_train_filtered=[]
y_train_filtered=[]


for i in range(28709):
    data=trainingdata.readline() #read one image
#    if data[0]=='0' or data[0]=='1' or data[0]=='2' or data[0]=='4':
#        continue
    image_string=data[3:-2] #cut off irrelevant part of the string
    image=np.array([int(k) for k in image_string.split(' ')])
    image=image.reshape(48,48)
    image=np.uint8(image)
    face_locations=face_recognition.face_locations(image)
    if i%1000==0:
        print(i)
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



x_test=x_train[-1000:-1]  #use last 1000 samples to test accuracy
x_train=x_train[0:-1000]
y_test=y_train[-1000:-1]
y_train=y_train[0:-1000]

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)


#convolutional neural network
model = tf.keras.models.Sequential()
model.add(Conv2D(64, (3,3) ,input_shape = x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(7))
model.add(Activation(tf.nn.softmax))

#simple neural network
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(7,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train,y_train, epochs=5)

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss,val_acc)

model.save('models/64,64,64,5.model') #save in models folder
