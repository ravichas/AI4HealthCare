# Course-1: Applying AI to 2D Medical Imaging Data
(*Please note that most of this section writeup was from Udacity and I have summarized portions of the writeup*)

## Project Overview
In this project, you will apply the skills that you have acquired in this 2D medical imaging course 
to analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray 
for the presence or absence of pneumonia. This project will culminate in a model that can predict 
the presence of pneumonia with human radiologist-level accuracy that can be prepared for 
submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission 
preparation, you will formally describe your model, the data that it was trained on, and a validation 
plan that meets FDA criteria.

You will be provided with the medical images with clinical labels for each image that were extracted 
from their accompanying radiology reports.

The project will include access to a GPU for fast training of deep learning architecture, as well 
as access to 112,000 chest x-rays (due to limit not included in this repo.) with disease labels acquired from 30,000 patients.

## Project Highlight
This project is designed to give you hands-on experience with 2D medical imaging data analysis 
and preparation of a medical imaging model for regulatory approval.

Upon completion of this project, you would be able to:

recommend appropriate imaging modalities for common clinical applications of 2D medical imaging
perform exploratory data analysis (EDA) on medical imaging data to inform model training and explain model performance
establish the appropriate ‘ground truth’ methodologies for training algorithms to label medical images
extract images from a DICOM dataset
train common CNN architectures to classify 2D medical images
translate outputs of medical imaging models for use by a clinician
plan necessary validations to prepare a medical imaging model for regulatory approval

## Starting the project

For this project, you will work in the Jupyter GPU workspace provided for you. You can also 
find the notebooks containing the necessary starter code in the workspace.

This repo contains: 

* `EDA.ipynb`: This is the file you will be performing the EDA.
* `Build and train model.ipynb`: This is the file you will be building and training your model.
* `Inference.ipynb`: This is the file you will be performing clinical workflow integration.
* `.dcm files`: They are the test files to test the clinical workflow integration.
* `sample_labels.csv`: This is the file that should be used to assess images in the pixel-level.
* `FDA_Submission_Template.md`: This is the template for you to create the FDA submission. 

Note: The NIH data for EDA and training is mounted in the Udacity Jupyter GPU workspace (not available for 
      anyone cloning this repo)  provided to you 
along with the code to load the data. Alternatively, you can download the data from the kaggle website
(https://www.kaggle.com/datasets/nih-chest-xrays/data) and 
run it locally. You are STRONGLY recommended to complete the project using the Udacity workspace since the 
data is huge, and you will need GPU to accelerate the training process.

## Project Steps
This project has the following steps.

* [Exploratory Data Analysis](EDA.ipynb)
* Building and Training Your Model
    * [Build and train model](Build%20and%20train%20model.ipynb)
    * [Inference](Inference.ipynb)
* [Clinical Workflow Integration, FDA Preparation](FDA_Submission_Template.md)


## Next steps suggested by Udacity team

* Create some of your own custom image augmentation (such as different image filtering techniques) rather than solely using those predefined by Keras’ ImageDataGenerator.

* Try creating two ‘nested’ models to specifically predict pneumonia. One that predicts pneumonia and/or infiltrates at the top level, and then a second model that specifically predicts pneumonia from the positive cases returned by the first model.

* Have your model output a class activation map in addition to a single binary prediction of pneumonia. This map will help a clinician to understand what the model is detecting as probable pneumonia in each image.

## Supporting links
* https://www.kaggle.com/datasets/nih-chest-xrays/data 
* You may also download all of the initial files for the project directly from this Udacity repo, https://github.com/udacity/AIHCND_C2_Starter 
* README file for the project is available from here, https://github.com/udacity/AIHCND_C2_Starter/blob/master/README.md


Acknowledgement: Udacity Nanodegree Team
