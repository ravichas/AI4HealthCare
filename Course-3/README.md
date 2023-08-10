# Course-3: Applying AI to Electronic Health Record (EHR) Data 

## Project Overview
EHR data is assuming a pivotal role as a primary source of real-world evidence (RWE) that drives decisions 
within the pharmaceutical industry and among regulators in shaping the course of clinical trials
(https://www.fda.gov/news-events/speeches-fda-officials/breaking-down-barriers-between-clinical-trials-and-clinical-care-incorporating-real-world-evidence). In this 
capacity, you find yourself donning the mantle of a data scientist within an exhilarating unicorn healthcare 
startup. This remarkable startup has given rise to an avant-garde diabetes treatment, poised for the rigors of 
clinical trial testing. The drug's uniqueness and sensitivity necessitate administration spanning a minimum of 
5 to 7 days within a hospital setting -a duration determined by dynamic factors embedded within the data's 
distribution. During this period, patients undergo frequent monitoring, testing, and receive adherence training
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3068890/) through a mobile application.

Leveraging a dataset furnished by a client partner, you're tasked with constructing a predictive model that 
can effectively pinpoint the type of patients meriting concentrated testing of this groundbreaking drug. The 
target demographic comprises individuals poised to remain hospitalized for the stipulated timeframe, without 
incurring substantial supplementary costs associated with drug administration and monitoring.

To accomplish this objective, your initial mission involves the development of a regression model capable of forecasting a patient's 
anticipated hospitalization period. Moreover, the model is expected to furnish an uncertainty range estimation, offering a spectrum of 
prediction confidence for subsequent ranking.

The Project Framework: The journey commences with the construction of a synthetic dataset-thoughtfully 
upsampled, denormalized, and endowed with line-level augmentation. This dataset is meticulously crafted 
upon the foundation of the UCI Diabetes readmission dataset. Within this ecosystem, students embark on 
the creation of a regression model that adeptly forecasts the expected duration of hospitalization, 
accompanied by a nuanced range of uncertainty estimation.

The Significance of Data Representation: This endeavor underscores the significance of building an apt data 
representation at the encounter level, underscored by judicious filtering, preprocessing, and feature 
engineering of vital medical code sets. As the journey unfolds, participants are called upon to dissect 
and interpret their models to identify biases across crucial demographic segments. Notably, the TF 
probability library is harnessed to impart uncertainty range estimates into the regression output 
predictions-a mechanism for prioritizing and stratifying prediction uncertainty levels.

Evaluating for Bias: At the culmination, the spotlight shifts to the creation of a demographic bias analysis. 
This analytical endeavor is poised to detect and evaluate potential biases residing within the model-an 
issue acutely pertinent in the context of healthcare data.

The project, in its entirety, stands as an illuminating demonstration of the intricate nexus between data 
science, healthcare, and predictive modeling -a symphony where innovation harmonizes with critical 
scrutiny to yield transformative insights.


(Note this section was taken from Udacity notes. Code developed was carried out in a VM with Jupyter environment
provided by Udacity. So, some of the instructions in this section may or may be applicable when 
you clone/use the code in this project. I have provided them for the sake of completion). 

## Context: 

EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators 
to make decisions on clinical trials. You are a data scientist for an exciting unicorn healthcare startup 
that has created a groundbreaking diabetes drug that is ready for clinical trial testing. It is a very 
unique and sensitive drug that requires administering the drug over at least 5-7 days of time in the 
hospital with frequent monitoring/testing and patient medication adherence training with a mobile 
application. You have been provided a patient dataset from a client partner and are tasked with building 
a predictive model that can identify which type of patients the company should focus their efforts testing 
this drug on. Target patients are people that are likely to be in the hospital for this duration of time 
and will not incur significant additional costs for administering this drug to the patient and monitoring.

In order to achieve your goal you must build a regression model that can predict the estimated hospitalization 
time for a patient and use this to select/filter patients for your study.

Expected Hospitalization Time Regression Model: Utilizing a synthetic dataset(denormalized at the 
line level augmentation) built off of the UCI Diabetes readmission dataset, students will build a 
regression model that predicts the expected days of hospitalization time and then convert this to 
a binary prediction of whether to include or exclude that patient from the clinical trial.

This project will demonstrate the importance of building the right data representation at the encounter level, 
with appropriate filtering and preprocessing/feature engineering of key medical code sets. This project 
will also require students to analyze and interpret their model for biases across key demographic groups.

## Dataset

Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available 
datasets and some datasets require training and approval. So, for the purpose of this exercise, 
we are using a dataset from UC Irvine that has been modified for this course. Please note that 
it is limited in its representation of some key features such as diagnosis codes which are 
usually an unordered list in 835s/837s (the HL7 standard interchange formats used for claims and remits).

https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008 
Data Schema The dataset reference information can be 
https://github.com/udacity/nd320-c1-emr-data-starter/tree/master/project/data_schema_references. 
There are two CSVs that provide more details on the fields and some of the mapped values.

## Prerequisites

* Intermediate level knowledge of Python
* Basic knowledge of probability and statistics
* Basic knowledge of machine learning concepts
* Installation of Tensorflow 2.0 and other dependencies(conda environment.yml or virtualenv requirements.txt 
  file provided)

## Environment Setup

For step by step instructions on creating your environment, please go to https://github.com/udacity/cd0372-Applying-AI-to-EHR-Data

## Learning Objectives

By the end of the project, you will be able to:

* Use the Tensorflow Dataset API to scalably extract, transform, and load datasets and build datasets 
  aggregated at the line, encounter, and patient data levels(longitudinal)
* Analyze EHR datasets to check for common issues (data leakage, statistical properties, 
  missing values, high cardinality) by performing exploratory data analysis.
* Create categorical features from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality 
  for high cardinality features by using embeddings
* Create derived features(bucketing, cross-features, embeddings) utilizing Tensorflow feature columns 
  on both continuous and categorical input features
* Use the Tensorflow Probability library to train a model that provides uncertainty range predictions 
  that allow for risk adjustment/prioritization and triaging of predictions
* Analyze and determine biases for a model for key demographic groups by evaluating performance 
  metrics across groups by using the Aequitas framework

## Steps to Completion

Please follow all of the directions in the Jupyter Notebook file in the classroom workspace 
or from the Github Repo if you decide to use your own environment to complete the project.

You complete the following steps there:
* Data Analysis
* Create Categorical Features with TF Feature Columns
* Create Continuous/Numerical Features with TF Feature Columns
* Build Deep Learning Regression Model with Sequential API and TF Probability Layers
* Evaluating Potential Model Biases with Aequitas Toolkit

The project submission notebook,student_project_submission.ipynb, is accessible from [here](student_project_submission.ipynb)

Acknowledgement: Udacity Nanodegree Team

