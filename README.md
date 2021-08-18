# Research Methodology Project (v 1.0)

## Introduction

This is the repository to detect American Sign Language using Computational Neural Network and Computer Vision. 

## Setup Environment

* Firstly, setup virtual environment to install necessary packages and libraries using the following command:

```
virtualenv .venv
source .venv/bin/activate
```

* To setup the environment use any Python based IDE (Pycharm for instance) and run the following command on the terminal:

`pip install -r requirements.txt`

* This command will install all the libraries and other reuirements necessary with the exact versions used in this project. You need to change the path of the dataset in the file `Collect_Data.py` by changing the `path` variable in the code.

```
path = 'C:/Pycharm/pythonProject/AslDetection/asldata'
```

* To save the CNN model, provide the desired path for the `model_trained.p` file in the write mode by changing the variable `pickle_out` in `Collect_Data.py` file.

```
pickle_out = open("C:\Pycharm\pythonProject\AslDetection\Settings\model_trained.p","wb")
```

## Project Flowchart

<img src="https://github.com/aryanvohra/RM_AslDetection/blob/main/Images/CNN_Architecture.png" align="center"  width="100" />

## Project Description

### Dataset

The dataset we have used for our model is known as ASL Alphabet Test which is an open source dataset and available at Kaggle. The dataset consists of a set of 780 images which are divided into 26 sub folders by the alphabets names (A-Z). Each image depicts the shape of the ASL gesture including some variations which are necessary to be trained for our model.The images consists of different light intensities, hand shapes, color and sizes to create a variation for our model. There are 30 images of each ASL alphabet from A to Z accounting to a total of 780 images. Each image is of 200 x 200 8-bit photos to match the asl-alphabet dataset. The data is structured in a folder format so that it is easy to use flow-from-directory in Tensorflow and Keras library of Python.

The dataset required for this project can be accessed from this link [ASL Data](https://github.com/aryanvohra/RM_AslDetection/tree/main/asldata). 

### Project Code Files

* #### Collect_Data.py

This file contains code for preprocessing, CNN model building, and training. We also save our CNN model in this code file which we will load in the file `CnnTest.py`

* #### CnnTest.py

In this code file, we load the CNN model created in the `Collect_Data.py` file and predict the accuracy of our mode and capture ASL-based alphabets from our video frame.

### Results

We tested our CNN model for three different gradient optimizers: `Adam` , `SGD` and `Adadelta`. Here are the results in terms of accuracy and loss.

Model    | Train Accuracy | Test Accuracy | Loss      | 
---      | ---            | ---           | ---       |            
Adam     | 0.944          |  0.5897       | 0.1645    |       
SGD      | 0.5827         |  0.3205       | 1.3414    |         
Adadelta | 0.8970         |  0.5008       | 0.3157    |           

As evident from the above table, Adam based optimizer fits best for our CNN model with the train and test accuracy of 0.944 and 0.525 respectively.


After fitting data in the model, we derive the ASL alphabets from the live video using OpenCv python library.


<img src="https://github.com/aryanvohra/RM_AslDetection/blob/main/Images/ASL_Symbol.png" align="center"  width="300" />

As visible from the top left corner, we have successfully predicted ASL alphabet `J` with a prediction of `0.997`


## Contact

You can contact Author for any support.
* __Email__: [avohra1@lakeheadu.ca](mailto:avohra1@lakeheadu.ca?subject=[GitHub]%20Source%20Han%20Sans)



