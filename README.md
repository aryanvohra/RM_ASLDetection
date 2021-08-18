# Research Methodology Project

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

* This command will install all the libraries and other reuirements necessary with the exact versions used in this project.
The dataset required for this project can be accessed from this link [ASL Data](https://github.com/aryanvohra/RM_AslDetection/tree/main/asldata). 
You need to change the path of the dataset in the file `Collect_Data.py` by changing the `path` variable in the code.

```
path = 'C:/Pycharm/pythonProject/AslDetection/asldata'
```

* To save the CNN model, provide the desired path for the `model_trained.p` file in the write mode by changing the variable `pickle_out` in `Collect_Data.py` file.

```
pickle_out = open("C:\Pycharm\pythonProject\AslDetection\Settings\model_trained.p","wb")
```

## Project Flowchart

<img src="https://github.com/aryanvohra/RM_AslDetection/blob/main/Images/CNN_Architecture.png" align="center"  width="100" />


