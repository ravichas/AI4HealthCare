# Course-1: Applying AI to 2D Medical Imaging Data

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

![PPG Sensor on Blood Flow](https://learn.udacity.com/nanodegrees/nd320/parts/cd0568/lessons/18f3eb82-4af0-40e3-8b53-7a8fd54c81da/concepts/f03327b1-fed6-4ac8-a0eb-981f48b5f2aeL)


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

This plot is created by computing the mean absolute error at all -- or at least 100 of -- the confidence 
thresholds in the dataset.

Building a confidence algorithm for pulse rate estimation is a little tricker than logistic regression 
because intuitively, there isn't some transformation of the algorithm output that can make a good confidence 
score. However, by understanding our algorithm behavior, we can come up with some general ideas that might 
create a good confidence algorithm. For example, if our algorithm is picking a strong frequency component 
that's not present in the accelerometer, we can be relatively confident in the estimate. Turn this idea 
into an algorithm by quantifying "strong frequency component".




Acknowledgement: Udacity Nanodegree Team







