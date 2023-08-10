# Course-4: Applying AI to wearable Device Data

(*Please note that most of this section writeup was from Udacity and I have summarized portions of the writeup*)

## Project Overview

This project has 2 primary components: 

1) Pulse Rate Algorithm Development:
Your journey commences with the development of a Pulse Rate Algorithm, drawing insights 
from the provided training data. Your task encompasses crafting an algorithm that accurately 
determines pulse rates. Subsequent to its creation, rigorous testing shall be conducted to validate 
that the algorithm satisfactorily fulfills predefined success criteria. This phase serves as the 
foundational cornerstone, showcasing your ability to translate data into actionable insights.

2) Clinical Application Integration and Healthcare Trends Discovery:
The second facet of this project delves into the practical application of your Pulse Rate Algorithm 
within a clinical context. Your algorithm is harnessed to compute a gamut of clinically meaningful 
features that extend beyond the realm of pulse rates. Within this arena, your mission transforms 
into a quest for healthcare trends—illuminating patterns, correlations, and insights that 
contribute to our understanding of healthcare dynamics. Through this exploration, you stand 
poised to uncover invaluable insights that have the potential to reshape medical practices and patient care.


## Introduction

A core feature that many users expect from their wearable devices is pulse rate estimation. Continuous pulse 
rate estimation can be informative for many aspects of a wearer's health. Pulse rate during exercise can be a 
measure of workout intensity and resting heart rate is sometimes used as an overall measure of cardiovascular 
fitness. In this project you will create a pulse rate estimation algorithm for a wrist-wearable device. 
Use the information in the Physiological Mechanics of Pulse Rate Estimation section below to inform the design of 
your algorithm. Make sure that your algorithm conforms to the given Algorithm Specifications in the following 
concept, Part 1: Overview & Instructions.

## Background

Physiological Mechanics of Pulse Rate Estimation
Pulse rate is typically estimated by using the PPG sensor. When the ventricles contract, the capillaries in the wrist fill with blood. The (typically green) light emitted by the PPG sensor is absorbed by red blood cells in these capillaries and the photodetector will see the drop in reflected light. When the blood returns to the heart, fewer red blood cells in the wrist absorb the light and the photodetector sees an increase in reflected light. The period of this oscillating waveform is the pulse rate.

![PPG Sensor on Blood Flow](https://video.udacity-data.com/topher/2020/February/5e3c882a_ppg-mechanics/ppg-mechanics.png)


## PPG Sensor on Blood Flow

However, the heart beating is not the only phenomenon that modulates the PPG signal. Blood in the wrist is fluid, 
and arm movement will cause the blood to move correspondingly. During exercise, like walking or running, we see 
another periodic signal in the PPG due to this arm motion. Our pulse rate estimator has to be careful not to 
confuse this periodic signal with the pulse rate.

We can use the accelerometer signal of our wearable device to help us keep track of which periodic signal is caused 
by motion. Because the accelerometer is only sensing arm motion, any periodic signal in the accelerometer is likely 
not due to the heart beating, and only due to the arm motion. If our pulse rate estimator is picking a frequency 
that's strong in the accelerometer, it may be making a mistake.

All estimators will have some amount of error. How much error is tolerable depends on the application. 
If we were using these pulse rate estimates to compute long term trends over months, then we may be more 
robust to higher error variance. However, if we wanted to give information back to the user about a 
specific workout or night of sleep, we would require a much lower error.

## Algorithm Confidence and Availability

Many machine learning algorithms produce outputs that can be used to estimate their per-result error. For 
example, in logistic regression, you can use the predicted class probabilities to quantify trust in the 
classification. A classification where 
one class has a very high probability is probably more accurate than 
one where all classes have similar probabilities. Certainly, this method is not perfect and won't perfectly 
rank-order estimates based on error. But if accurate enough, it allows consumers of the algorithm more 
flexibility in how to use it. We call this estimation of the algorithm's error the confidence.

In pulse rate estimation, having a confidence value can be useful if a user wants just a handful of 
high-quality pulse rate estimate per night. They can use the confidence algorithm to select the 
20 most confident estimates at night and ignore the rest of the outputs. Confidence estimates 
can also be used to set the point on the error curve that we want to operate at by sacrificing the 
number of estimates that are considered valid. There is a trade-off between availability and error. 
For example, if we want to operate at 10% availability, we look at our training dataset to determine 
the confidence threshold for which 10% of the estimates pass. Then if only if an estimate's confidence 
value is above that threshold, do we consider it valid. See the error vs. availability curve below.

![MAE vs. Availability](https://video.udacity-data.com/topher/2020/February/5e3c8829_error-vs-availability/error-vs-availability.png)


This plot is created by computing the mean absolute error at all -- or at least 100 of -- the confidence 
thresholds in the dataset.

Building a confidence algorithm for pulse rate estimation is a little tricker than logistic regression 
because intuitively, there isn't some transformation of the algorithm output that can make a good confidence 
score. However, by understanding our algorithm behavior, we can come up with some general ideas that might 
create a good confidence algorithm. For example, if our algorithm is picking a strong frequency component 
that's not present in the accelerometer, we can be relatively confident in the estimate. Turn this idea 
into an algorithm by quantifying "strong frequency component".


## Part 1: Overview & Instructions

## Part 1: Pulse Rate Algorithm Project Overview

### Algorithm Specifications

You must build an algorithm that:

* estimates pulse rate from the PPG signal and a 3-axis accelerometer.
* assumes pulse rate will be restricted between 40BPM (beats per minute) and 240BPM
* produces an estimation confidence. A higher confidence value means that this estimate should be more 
  accurate than an estimate with a lower confidence value.
* produces an output at least every 2 seconds.

### Success Criteria
Your algorithm performance success criteria are as follows: the mean absolute error at 90% availability 
must be less than 15 BPM on the test set. Put another way, the best 90% of your estimates--according to 
your own confidence output-- must have a mean absolute error of less than 15 BPM. The evaluation function 
is included in the starter code.

Note that you will not have access to the dataset that the unit test will call `AggregateErrorMetric` on the 
output of your `RunPulseRateAlgorithm` on a test dataset that you do not have access to. The result of this 
call must be less than 15 BPM for your algorithm's performance to pass. The test set should be easier 
than the training set, so as long as your algorithm is doing reasonably well on the training data set, 
it should pass this test.

## Some Helpful Tips

1. Remember to bandpass filter all your signals. Use the 40-240BPM range to create your pass band.
2. Use `plt.specgram` to visualize your signals in the frequency domain. You can plot your estimates on 
  top of the spectrogram to see where things are going wrong.
3. When the dominant accelerometer frequency is the same as the PPG, try picking the next strongest PPG 
  frequency if there is another good candidate.
4. Sometimes the cadence of the arm swing is the same as the heartbeat. So if you can't find another 
 good candidate pulse rate outside of the accelerometer peak, it may be the same as the accelerometer.
5. One option for a confidence algorithm is to answer the question, "How much energy in the frequency 
spectrum is concentrated near the pulse rate estimate?" You can answer this by summing the frequency 
spectrum near the pulse rate estimate and dividing it by the sum of the entire spectrum.

### Dataset
You will be using the Troika[1] dataset to build your algorithm. Find the dataset under 
  `datasets/troika/training_data` (will not be included in this repo). 
  The README in that folder will tell you how to interpret the data. The starter code contains a 
  function to help load these files.

1. Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart 
   Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical 
   Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. 
   [Link](https://ieeexplore.ieee.org/document/6905737)

### Getting Started

The starter code includes a few helpful functions. `TroikaDataset`, `AggregateErrorMetric`, and `Evaluate` 
do not need to be modified.

1. Use `TroikaDataset` to retrieve a list of .mat files containing reference and signal data.
2. Use `scipy.io.loadmat` to the .mat file into a python object.
3. The bulk of the code will be in the `RunPulseRateAlgorithm` function. You can and should break the 
   code out into multiple functions.
4. `RunPulseRateAlgorithm` will take in two filenames and return a tuple of two 
    NumPy arrays--per-estimate pulse rate error and confidence values. 
    Note: Remember to write docstrings for all functions that you write (including `RunPulseRateAlgorithm`)
5. Finally, run the Evaluate function to call your algorithm on the Troika dataset and compute an aggregate 
   error metric. Hint: While building the algorithm, you may want to inspect the algorithm errors in more detail.

### Instructions

Offline Instructions

1. Clone the project starter repo [here](https://github.com/udacity/nd320-c4-wearable-data-project-starter)..
2. Open up the pulse_rate_starter.ipynb in a local program (e.g., spyder IDE, Jupyter Notebooks, etc. 
   which you can find more detailed instructions in the Introduction to Wearables Lesson's Developer's 
   Workflow Concept.)
3. You are ready to begin Part 1 of the Final Project.

Notes: It is good to confirm that the packages scipy is version 1.2.0+. 
       You can do this by opening a new terminal and typing conda list and confirm the versioning is appropriate.

### Online Instructions

You should head to the next Concept Pulse Rate Algorithm.
Confirm that the pulse_rate_starter.ipynbis open and you are ready to complete Part 1 of the Final Project.





Acknowledgement: Udacity Nanodegree Team







