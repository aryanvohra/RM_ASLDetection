# Importing libraries
import numpy as np
import cv2
import pickle

# Assigning frame width and height
w,h = 640,480
threshold = 0.65
capture = cv2.VideoCapture(0)
character = ""

capture.set(3,w)
capture.set(4,w)

# Loading the saved model
pickle_input = open("C:\Pycharm\pythonProject\AslDetection\Settings\model_trained.p","rb")
model = pickle.load(pickle_input)

# Preprocessing the captured video frames
def preProcessing(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  img = img/255
  return img

# Capturing the preprocessed image
while True:
    success, orig_img = capture.read()
    img = np.asarray(orig_img)
    img = cv2.resize(img,(40,40))
    img = preProcessing(img)
    cv2.imshow("processed image", img)
    img = img.reshape(1,40,40,1)

    # Predicting the image
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    prob_val =np.amax(predictions)

    # Comparing predicted value with the thresold
    # Saving the ASL alphabets comapring it to classIndex.
    if prob_val>threshold:
        if classIndex ==0:
            character="A"
        if classIndex ==1:
            character="B"
        if classIndex ==2:
            character="C"
        if classIndex ==3:
            character="D"
        if classIndex ==4:
            character="E"
        if classIndex ==5:
            character="F"
        if classIndex ==6:
            character="G"
        if classIndex ==7:
            character="H"
        if classIndex ==8:
            character="I"
        if classIndex ==9:
            character="J"
        if classIndex ==10:
            character="K"
        if classIndex ==11:
            character="L"
        if classIndex ==12:
            character="M"
        if classIndex ==13:
            character="N"
        if classIndex ==14:
            character="O"
        if classIndex ==15:
            character="P"
        if classIndex ==16:
            character="Q"
        if classIndex ==17:
            character="R"
        if classIndex ==18:
            character="S"
        if classIndex ==19:
            character="T"
        if classIndex ==20:
            character="U"
        if classIndex ==21:
            character="V"
        if classIndex ==22:
            character="W"
        if classIndex ==23:
            character="X"
        if classIndex ==24:
            character="Y"
        if classIndex ==25:
            character="Z"

        cv2.putText(orig_img,character+" "+str(prob_val),(50,50),
                    cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)


    # Displaying captured video with recognized ASL alphabets and accuracy
    cv2.imshow("Original Img", orig_img)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

