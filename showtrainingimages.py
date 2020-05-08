import numpy as np
from matplotlib import pyplot as plt

trainingdata = open("2915trainingimages.txt","r")
trainingdata.readline() #skip first line

emotions=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

for i in range(10):
    data=trainingdata.readline() #read one image
    emotion=emotions[int(data[0])]
    image_string=data[3:-2] #cut off irrelevant part of the string
    image = np.array([int(k) for k in image_string.split(' ')])
    image=image.reshape((48,48))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    print(emotion)
    
trainingdata.close()

