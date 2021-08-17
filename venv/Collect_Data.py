# Imorting libraries
import keras
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,InputLayer
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# Loading the dataset
# Specify the dataset path in the path variable
path = 'C:/Pycharm/pythonProject/AslDetection/asldata'

images = []
classcount = 0
classNumber = []

# Reading dataset
path_list = os.listdir(path)
path_list.sort()
categories = len(path_list)

# Iterating alphabets folder
for i in path_list:
  alphabets_fold = os.listdir(path+'/'+str(i))
  classcount = classcount+1
  for j in alphabets_fold:
    # iterate in image list
    current_img = cv2.imread(path+'/'+str(i)+'/'+j)
    #reduce size for lower computational process
    current_img = cv2.resize(current_img,(40,40))
    images.append(current_img)
    classNumber.append(classcount)

images = np.array(images)
classNumber= np.array(classNumber)

# Splitting DataSet
X_train,X_test,y_train,y_test = train_test_split(images,classNumber,test_size = 0.20)
#X_train,X_validation,y_test,y_validation = train_test_split(X_train,y_train,test_size = 0.20,train_size=0.80)


#image preprocessing function
def preProcessing(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = img/255
  return img


# preprocessing train data
X_train = np.array(list(map(preProcessing,X_train)))

# preprocessing test data
X_test = np.array(list(map(preProcessing,X_test)))

# reshaping the train data into single column
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)

# reshaping the test data into single column
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


numSamples=[]
for x in range(0,classcount):
  numSamples.append(len(np.where(y_train==x)[0]))

# Plotting graph of total number of images in each class
plt.figure(figsize=(10,5))
plt.bar(range(0,classcount),numSamples)
plt.title('Number of images in each class')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.show()
# Converting images into vectors
dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,
                             shear_range=0.1,rotation_range=10)

dataGen.fit(X_train)
# Dividing data into unique categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Initializing the model
def CnnModel():
  model = keras.Sequential()
  model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(40, 40, 1)))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Dropout(0.5))
  model.add(Flatten())

  model.add(Dense(250, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(27, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model

# Calling model function
model = CnnModel()
model.summary()

# Training the model
fit_model = model.fit(X_train,y_train,steps_per_epoch=20,epochs=10)

# Evaluating the model on test data
accuracy= model.evaluate(X_test,y_test)
print("here is the accuracy", accuracy)

# Saving model using pickle
pickle_out = open("C:\Pycharm\pythonProject\AslDetection\Settings\model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

