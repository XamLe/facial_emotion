#!/usr/bin/env python
# coding: utf-8

# In[12]:


import csv
import pandas

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


"""
with open('./../Resources/dataset/fer2013.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
            
"""


# In[14]:


file_name = './../Resources/dataset/fer2013.csv'
# df = pandas.read_csv(file_name)
# print(df)


# In[15]:


def getData():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    first = True

    for line in open(file_name):
        if first:
            first = False
        else:
            row = line.split(',')
            if row[2] == "Training\n":
                Y_train.append(int(row[0]))
                X_train.append([int(p) for p in row[1].split()])
            else:
                Y_test.append(int(row[0]))
                X_test.append([int(p) for p in row[1].split()])

    X_train = np.asarray(X_train, np.float32)
    X_test = np.asarray(X_test, np.float32)
    X_train = X_train / 255
    X_test = X_test / 255
    X1 = []
    for idx in range(len(X_train)):
        X1.append(tf.reshape(tf.convert_to_tensor(X_train[idx], np.float32), [48,48]))
    X_train = X1
    X1 = []
    for idx in range(len(X_test)):
        X1.append(tf.reshape(tf.convert_to_tensor(X_test[idx], np.float32), [48,48]))
    X_test = X1

    Y_train = tf.convert_to_tensor(Y_train)
    Y_test = tf.convert_to_tensor(Y_test)

    return X_train, Y_train, X_test, Y_test


# In[16]:


X_train, Y_train, X_test, Y_test = getData()


# In[11]:


# plt.imshow(X_test[1])
# Y_test[1]


# In[ ]:





# In[ ]:




