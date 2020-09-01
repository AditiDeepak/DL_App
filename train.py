from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from skimage import io
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils.vis_utils import plot_model
from skimage import io
from skimage.transform import resize, rescale, rotate, setup, warp, AffineTransform
import pandas as pd
from sklearn.metrics import confusion_matrix



class net:
    @staticmethod
    def build(height,width,depth,num_classes):#Number of channels-1(grayscale),3(RGB)
        model=Sequential()
        shape=(height,width,depth)
        #Channel last ordering is default for tensorflow
        if K.image_data_format()=="channels_first":
            shape=(depth,height,width)
        #To add layers for model->Layer1
        model.add(Conv2D(30,(5,5),padding="same",input_shape=shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))#Moves 2 steps along x and y directions
        #Layer2
        model.add(Conv2D(60,(5,5),padding="same"))#The same padding means zero padding is provided,whereas in VALID->No zero padding,values are dropped 
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #FUlly connected layer->We flatten out our inputs
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        #Softmax classifier layer->This must have the same number of nodes as the output layer
        #Two types of softmax ->Full softmax(is good when the dataset is small),Candidate softmax(Better for larger sets)
        #This will yield the probability of each class
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))
        return model

total_epoch=50
learning_rate=0.001
batch_size=50

data=[]
labels=[]
temp=[]

for root,sub,files in os.walk('G:\pro'):
    for name in files:
        num=os.path.join(root,name)
        data.append(num)
    
for image in data:
    im1=io.imread(image)
    im2=resize(im1,(224,224))
    im3=img_to_array(im2)
    temp.append(im3)
#temp contains images
for path in data:
    path=path.split(os.path.sep)[-2]
    if path=='India Gate':
        label=1
        labels.append(label)
    elif path=='Qutub Minar':
        label=2
        labels.append(label)
    elif path=='Taj Mahal':
        label=3
        labels.append(label)

#labels contains labels

#After obtaining images and labels,split them into train and test
#scale the intensities to [0,1]
temp_array=np.array(temp,dtype="float")/255.0
label_array=np.array(labels)
(trainX,testX,trainY,testY)=train_test_split(temp_array,label_array,test_size=0.20,random_state=42)
#Convert integers to vectors
trainY= pd.get_dummies(trainY).values
testY=pd.get_dummies(testY).values#get them in the form of one hot labels in array
#matrix multiplication with np.zeros

#Next is data augmentation-TO increase the number of samples
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print('Compiling model....')
model=net.build(height=224,width=224,depth=3,num_classes=3)
opt=Adam(lr=learning_rate,decay=learning_rate/total_epoch)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

print("training network..........")
train_test_fit=model.fit_generator(aug.flow(trainX,trainY,batch_size=batch_size),validation_data=(testX,testY),steps_per_epoch=len(trainX)//batch_size,epochs=total_epoch,verbose=1)

model.summary()#To give the summary of each of the layers.
test_pred=model.predict(testX)
test_pred=(test_pred>0.5)
cm=confusion_matrix(testY.argmax(axis=1),test_pred.argmax(axis=1))


