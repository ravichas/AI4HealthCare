# Course-2: Applying AI to 3D Medical Imaging Data
(*Please note that most of this section writeup was from Udacity and I have summarized portions of the writeup*)

## Project Overview

### Quantification of Hippocampus Volume in Alzheimer's Disease Progression

## Introduction
Alzheimer's disease (AD) stands as a progressive neurodegenerative condition characterized by compromised 
neuronal function leading to eventual cell death. This disorder represents the leading cause of dementia, 
marked by clinical indicators such as memory loss, diminished capacity for new learning, language function 
decline, and other discernible manifestations.

For individuals presenting initial symptoms, the ability to quantify disease advancement over time 
holds significant potential in guiding therapeutic interventions and disease management strategies.

Among the various techniques available, radiological assessments utilizing MRI examinations stand out 
as a sophisticated means of quantifying disease progression. In particular, the measurement of hippocampal 
volume has emerged as a valuable tool for diagnosing and monitoring the progression of numerous brain 
disorders, prominently including AD. Substantiated by studies, a notable reduction in hippocampal volume 
has been consistently observed in AD patients.

The hippocampus, a pivotal structure within the human brain (as well as in the brains of other vertebrates), 
plays a critical role in consolidating information from short-term memory into long-term memory. 
In essence, the hippocampus is widely recognized as the center responsible for memory and learning 
processes, forming the foundation for our cognitive abilities.

<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813bf9_hippocampus-small/hippocampus-small.gif">
<p> Source: Life Science Databases (LSDB). Hippocampus. Images are from Anatomography maintained by
    LSDB. (2010). CC-BY-SA 2.1jp.[Link](https://commons.wikimedia.org/wiki/File:Hippocampus_small.gif) 
</p>
</div>


Humans possess a pair of hippocampi, with one situated within each hemisphere of the brain. 
These vital structures are situated in the medial temporal lobe. An intriguing tidbit - the 
term "hippocampus" finds its origins in Greek, approximately translating to "horselike," 
owing to its resemblance to a seahorse, as noted by one of the early anatomists who 
depicted the structure. This analogy becomes apparent upon viewing the accompanying image.`

<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813bf7_hippocampus-and-seahorse-cropped/hippocampus-and-seahorse-cropped.jpg">
<p> Seahorse & Hippocampus 
    Source: Seress, Laszlo. Laszlo Seress preparation of a human hippocampus along with a sea horse 
    (1980). CC-BY-SA 1.0.[Link](https://commons.wikimedia.org/wiki/File:Hippocampus_and_seahorse.JPG) 
</p>
</div>

As noted by Nobis et al. (2019; https://www.sciencedirect.com/science/article/pii/S2213158219302542)),the hippocampal volume exhibits variation within a population, encompassing 
specific parameters and defined limits. It becomes feasible to establish a "normal" range by considering 
factors such as age, sex, and brain hemisphere.

This phenomenon is visually demonstrated in the accompanying image, illustrating the right 
hippocampal volume of women spanning ages 52 to 71.


<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813c01_nomogram-fem-right/nomogram-fem-right.jpg">
<p> Nanogram - Female, Right Hippocampus Volume, Corrected for head Size, 
    Source: Nobis, L., Manohar, S.G., Smith, S.M., Alfaro-Almagro, F., Jenkinson, M., Mackay, C.E., Husain, M. Hippocampal volume across age: Nomograms derived from over 19,700 people in UK Biobank. Neuroimage: Clinical, 23(2019), pp. 2213-1582.
</p>
</div>

However, there exists a challenge in measuring hippocampal volume through MRI scans. The process can be 
rather laborious, as each individual slice of the 3D volume requires meticulous analysis 
and the structure's contours must be traced. Complicating matters further, the hippocampus 
exhibits a non-uniform shape.

Can you discern the hippocampi in the provided axial slice below?

<div align="center">
<img src="https://video.udacity-data.com/topher/2020/April/5e95548d_mri/mri.jpg">
<p> Axial slice of an MRI image of the brain </p>
</div>

By now, you might have surmised our intention to develop an AI software designed to assist clinicians 
in accomplishing this task with greater speed and consistency.

Throughout this course, you've observed that a significant portion of AI development involves 
the meticulous curation of datasets and the demonstration of clinical effectiveness. In this project, 
our primary emphasis will be directed toward the technical facets of constructing a segmentation model 
and seamlessly incorporating it into the clinician's operational process. As such, we'll leave 
the aspects of dataset curation and extensive model validation somewhat beyond 
the immediate scope of this project.

## What you will build? 

Within this undertaking, you will orchestrate the development of a comprehensive AI 
system. This system will encompass a machine learning algorithm seamlessly incorporated 
into a clinical-grade viewer, its prime function being the automatic computation of 
hippocampal volumes for newly presented patients. This computation will take place 
as their imaging studies are committed to the clinical imaging archive.

An advantageous aspect is that you need not contend with entire patient cranial scans. 
To streamline the process, our (fictional) radiology department employs a HippoCrop tool, 
which adeptly extracts a rectangular region from each brain scan image series. This thoughtful implementation 
will ease your task considerably. Furthermore, our diligent radiologists have meticulously curated 
and annotated a dataset containing pertinent volumes. Remarkably, they've even converted these annotations 
to the NIFTI format.

The dataset at your disposal contains segmentations of the right hippocampus, and your endeavor 
will involve leveraging the U-Net architecture to construct the segmentation model.

Upon successfully achieving this, your focus will shift to the seamless integration of this model 
into a functional clinical PACS. This integration aims to ensure that the model becomes an intrinsic 
component of the workflow, operating on every incoming study and generating comprehensive 
volume measurement reports.


## Dataset

We will be using "Hippocampus" dataset from the [Medical Decathlon Competition](medicaldecathlon.com). 
This dataset adopts a configuration of NIFTI files, each corresponding to a distinct volume, along with 
an accompanying segmentation mask. The original images comprising this dataset consist of T2 MRI 
scans encompassing the entirety of the brain.

It's noteworthy that, within this dataset, our focus is on cropped volumes. Specifically, these 
volumes exclusively encapsulate the region surrounding the hippocampus. This strategic cropping approach 
has a twofold impact: firstly, it substantially reduces the dataset's overall size; secondly, it 
streamlines the complexity of our machine learning task, thereby affording us more feasible 
training durations. It's essential to underscore that this doesn't undermine the significance 
of our challenge—it's far from being considered a "toy" problem. In the realm of medical imaging, 
algorithms designed to crop rectangular regions of interest are commonly employed. It's important to 
acknowledge that segmentation remains an intricate undertaking, demanding rigorous attention.

## Programming Environment

We have two options to complete this project

## Option-1: Udacity Workspaces: 
This will be a self-contained environment that one can run directly on their web-browser. 
This option was used to complete the project. 

## Option-2 Local Environment (sharing the details, just for the sake of completion)
Need the follogin libraries for the first two sections of the project

* Python 3.7+ environment
* PyTorch (preferably with CUDA)
* nibabel (https://nipy.org/nibabel/)
* matplotlib
* numpy
* pydicom (https://pydicom.github.io/pydicom/stable/tutorials/installation.html)
* pillow (should be installed with pytorch)
* tensorboard (https://pypi.org/project/tensorboard/)

In the third section of the project we will be working with three software products for emulating the clinical
network. You should need to install and configure 

* [Orthanc Server](https://www.orthanc-server.com/download.php) for PACS emulation 
* [OHIF zero-footprint web viewer](https://docs.ohif.org/development/getting-started)
for viewing images. Note that if you deploy OHIF from its github repository, 
at the moment of writing the repo includes a yarn script (`orthanc:up`) where it downloads 
and runs the Orthanc server from a Docker container. If that works for 
you, you won't need to install Orthanc separately.
* If you are using Orthanc (or other DICOMWeb server), you will need to configure OHIF to read data 
from your server. OHIF has instructions for this: https://docs.ohif.org/platform/extensions/modules/data-source
* In order to fully emulate the Udacity workspace, you will also need to configure Orthanc for 
auto-routing of studies to automatically direct them to your AI algorithm. For this you will 
need to take the script that you can find at `section3/src/deploy_scripts/route_dicoms.lua`
and install it to Orthanc as explained on this page: https://book.orthanc-server.com/users/lua.html
* [DCMTK tools](https://dcmtk.org/en/)for testing and emulating a modality. Note that if you 
are running a Linux distribution, you might be able to install dcmtk directly from 
the package manager (e.g. `apt-get install dcmtk` in Ubuntu)


## Section 1: Curating a Dataset of Brain MRIs

<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813c00_slicer/slicer.png">
<p> Slicer image </p>
</div>

You will perform this operation in *Workspace for Section 1**. This workspace (Udacity) has a 
Python virual environment called **medai** which is setup with everything that you need to 
train your ML model.

The data is located in the VM at `/data/TrainingSet` folder [here](https://github.com/udacity/nd320-c3-3d-imaging-starter/tree/master/data/TrainingSet)

In the designated project directory labeled `section1,` you will locate a Python Notebook. This 
Notebook is thoughtfully equipped with a set of instructions aimed at assisting you in scrutinizing 
the dataset, enhancing your comprehension of the clinical aspects of the challenge, and preparing 
the data for seamless integration with your algorithm in Section 2. Within this notebook, you 
will encounter two distinct types of comments:

Comments annotated with `# TASK`: signify explicit tasks, instructions, or inquiries that require 
your attention and completion.

Comments lacking the aforementioned annotation are optional in nature. However, they serve 
as suggestions, queries, or contextual information that is designed to enrich your grasp 
of the subject matter. Engaging with these supplementary comments can contribute to a 
more profound understanding of the domain and facilitate the application of your newfound skills 
in exploring medical imaging datasets.


## Instructions

Once you complete the tasks, copy the following to the directory `section1/out`: 
1. Curated dataset with labels, as collection of NIFTI files. Amount of training image 
volumes should be the same as the amount of label volumes`
2. A Python notebook (or a raw python file) with the results (plus comments) of 
   the Exploratory Data Analysis. 

Note: You can launch Jupyter running the following bash script, `bash launch_jupyter.sh`

Please find `Section 1` code/data files [here](section1.tar.gz). Here is a tree diagram of the contents of 
section1 
```
section1
`-- out
    |-- Final\ Project\ EDA.ipynb
    |-- README.md
    |-- images
    |   |-- hippocampus_001.nii.gz
    |   |-- hippocampus_003.nii.gz
    |   |-- hippocampus_004.nii.gz
    |   |-- hippocampus_006.nii.gz
    |   |-- hippocampus_007.nii.gz
    |   |-- ......................
    |   |-- hippocampus_383.nii.gz
    |   |-- hippocampus_385.nii.gz
    |   |-- hippocampus_386.nii.gz
    |   |-- hippocampus_387.nii.gz
    |   |-- hippocampus_389.nii.gz
    |   |-- hippocampus_390.nii.gz
    |   |-- hippocampus_393.nii.gz
    |   `-- hippocampus_394.nii.gz
    |-- img
    |   |-- Slicer.png
    |   `-- nomogram_fem_right.svg
    `-- labels
        |-- hippocampus_001.nii.gz
        |-- hippocampus_003.nii.gz
        |-- hippocampus_004.nii.gz
        |-- hippocampus_006.nii.gz
        |-- hippocampus_007.nii.gz
        |-- hippocampus_008.nii.gz
        |-- hippocampus_011.nii.gz
        |-- ......................
        |-- hippocampus_381.nii.gz
        |-- hippocampus_383.nii.gz
        |-- hippocampus_385.nii.gz
        |-- hippocampus_386.nii.gz
        |-- hippocampus_387.nii.gz
        |-- hippocampus_389.nii.gz
        |-- hippocampus_390.nii.gz
        |-- hippocampus_393.nii.gz
        `-- hippocampus_394.nii.gz

5 directories, 524 files
```

## Section 2: Training a Segmentation CNN


<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813bfb_loss/loss.png">
<p> 
</p>
</div>

In this section you will perform the following in a **Workspace 
for Section 2**. This workspace will have the same environment as 
**Workspace for Section 1**

in the directory called `home/src` you will find the source 
code that forms the framework for your machine leaning pipeline. 

You will be using `PyTorch` to train the model, similar to the
segmentation & Classification lesson and `Tensorboard` to visualize 
the results.

Here are the links for `PyTorch` and `Tensorboard`
`PyTorch`: https://pytorch.org
`Tensorboard`: https://tensorflow.org/tensorboard/ 

You will use the script `run_ml_pipelilne.py` to start off the
training pipeline. Before you run the scirpt, fix the code
lines that contain `# TASK`

When you are ready, run the code `run_ml_pipeline.py`
The code has hooks to log progress to Tensorboard. In order to see 
the Tensorboard output you need to launch Tensorboard executable from the same directory where `run_ml_pipeline.py` is located using the following command: 

`cd src`
`tensorboard --logdir runs --bind-all`

After this step, tensorboard will write logs into directory called
`runs` and you will be able to view progress by opening the browser 
and navigating to the default port # 6006 of the machine you are running
it (ex., `http://f819a6ac7f2cc:6006/`).


Instructions

Once you complete this section, copy the files to the directory `section2/out`: 
1) Functional code that trains the segmentation model
2) Test report with DICE scores on test set (json file). Your average Dice with def model should around .90
3) screen shots from Tensorboard output, shown the loss plots 
4) Trained model (model.pth) (deleted due to space limitation)


Please find `Section 2` code/data files [here](section2.tar.gz). Here is a tree diagram of the contents of 
section2 
```
home
|-- launch_jupyter.sh
|-- section2
|   `-- out
|       |-- Annotation\ 2023-03-18\ 231845.png
|       |-- ImageData_step99.png
|       |-- Prediction_step99.png
|       |-- ProbabilityMap_step99.png
|       |-- README.txt
|       |-- loss.png
|       |-- model.pth
|       `-- results.json
`-- src
    |-- data_prep
    |   |-- HippocampusDatasetLoader.py
    |   |-- SlicesDataset.py
    |   `-- __pycache__
    |       |-- HippocampusDatasetLoader.cpython-38.pyc
    |       `-- SlicesDataset.cpython-38.pyc
    |-- environment.yml
    |-- experiments
    |   |-- UNetExperiment.py
    |   `-- __pycache__
    |       |-- UNetExperiment.cpython-36.pyc
    |       `-- UNetExperiment.cpython-38.pyc
    |-- inference
    |   |-- UNetInferenceAgent.py
    |   |-- __init__.py
    |   `-- __pycache__
    |       |-- UNetInferenceAgent.cpython-38.pyc
    |       `-- __init__.cpython-38.pyc
    |-- networks
    |   |-- RecursiveUNet.py
    |   |-- __init__.py
    |   `-- __pycache__
    |       |-- RecursiveUNet.cpython-38.pyc
    |       `-- __init__.cpython-38.pyc
    |-- run_ml_pipeline.py
    |-- runs
    |-- utils
    |   |-- __pycache__
    |   |   |-- utils.cpython-38.pyc
    |   |   `-- volume_stats.cpython-38.pyc
    |   |-- utils.py
    |   `-- volume_stats.py
    `-- utils.tar.gz
```






## Section 3: Integrating into a clinical network

<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813bff_ohif/ohif.png">
<p> 
</p>
</div>

In this final section you will use some of the work you did for Section 2 to create an AI product that can be integrated into a clinical network and provide the auto-computed information on the hippocampal volume to the clinicians. While hospital integrations are typically handled by hospital IT staff, it will help tremendously if you can talk the same language with the people who will operate your model, and will have a feel for how clinical radiological software works. These skills will also help you debug your model in the field.

You will perform this section in a different workspace than the previous two sections: Workspace for Section 3. This workspace is a simpler hardware, with no GPU, which is more representative of a clinical environment. This workspace also has a few tools installed in it, which is replicates the following clinical network setup:


<div align="center">
<img src="https://video.udacity-data.com/topher/2020/March/5e813bfd_network-setup/network-setup.png">
<p> 
</p>
</div>


Specifically, we have the following software in this setup:

* MRI scanner is represented by a script `section3/src/deploy_scripts/send_volume.sh`. When you run this script 
it will simulate what happens after a radiological exam is complete, and send a volume to the clinical PACS. 
Note that scanners typically send entire studies to archives. *PACS server is represented by Orthanc 
deployment that is listening to DICOM DIMSE requests on port 4242. Orthanc also has a DicomWeb interface 
that is exposed at port 8042, prefix /dicom-web. There is no authentication and you are welcome to 
explore either one of the mechanisms of access using a tool like curl or Postman. Our PACS server is 
also running an auto-routing module that sends a copy of everything it receives to an AI server. 
See instructions ad the end of this page on how to launch if you are using the Udacity Workspace.

* Viewer system is represented by OHIF. It is connecting to the Orthanc server using DicomWeb and is serving a 
web application on port 3000. Again, see instructions at the end of this page if you are using the Udacity Workspace.

* AI server is represented by a couple of scripts. `section3/src/deploy_scripts/start_listener.sh` brings 
up a DCMTK's `storescp` and configures it to just copy everything it receives into a directory that you 
will need to specify by editing this script, organizing studies as one folder per study. HippoVolume.
AI is the AI module that you will create in this section.

If you want to replicate this environment on your local machine, you will find instructions in the 
Project Overview concept.

As with Section 2, in the directory called section3/src you will find the source code that forms the 
skeleton of the HippoVolume.AI module.

`inference_dcm.py` is the file that you will be working on. It contains code that will analyze the directory 
of the AI server that contains the routed studies, find the right series to run your algorithm on, 
will generate report, and push it back to our PACS.

Note that in real system you would architect things a bit differently. Probably, AI server would be a 
separate piece of software that would monitor the output of the listener, and would manage multiple 
AI modules, deciding which one to run, automatically. In our case, for the sake of simplicity, all code 
sits in one Python script that you would have to run manually after you simulate an exam 
via the send_volume.sh script - inference_dcm.py. It combines the functions 
of processing of the listener output and executing the model, and it does not do 
any proper error handling :)

As before, you will need to follow the instructions inside the code files to complete the section 
and create your AI module. Same convention is used as in Sections 1 and 2: comments that start 
with # TASK instruct you to create certain code snippets, and all other types of comments provide 
background or stand-out suggestions.

You will need to complete all the instructional comments in the code in order 
to complete this section. You can do this in any order, but it makes most sense to start with the code 
in inference_dcm.py.

Once you complete the code, you can test it by running

deploy_scripts/send_volume.sh

which will simulate a completion of MRI study and sending of patient data to our PACS, 
and then following that by running inference_dcm.py

The send_volume.sh script needs to be run from directory section3/src (because it 
relies on relative paths). If you did everything correctly, an MRI scan will be sent 
to the PACS and to your module which will compute the volume, prepare the report and push 
it back to the PACS so that it could be inspected in our clinical viewer.

At this point, go to [YOUR IP ADDRESS]:3000 (can be another port if you are using Udacity Workspace) 
which brings up our OHIF viewer. You should be able to inspect your report in all its glory, in the 
context of a radiological study presented to a radiologist in a clinical viewer!

The study that send_result.sh sends, and a few other sample studies are located in /data/TestVolumes. 
Feel free to modify the script to try out your algorithm with other volumes.

Note, that the DICOM studies used for inferencing this section have been created artificially, 
and while full-brain series belong to the same original study, this is not the study from 
which the hippocampus crop is taken.

Now that you have built a radiological AI system and given it to clinicians, you can start collecting 
data on how your model performs in the real world. If you (or the company you work for) intends to 
commercialize your technology, you will have to clear the regulatory bar. As we have discussed in our 
final lesson, an important contribution of an AI engineer to this endeavor is helping execute the 
clinical validation by contributing to a validation plan. Your final task in this course is to 
write a draft of such plan (shoot for 1-2 pages for this exercise). Remember - clinical validation 
is all about proving that your technology performs the way you claim it does. If you are saying that 
it can measure hippocampal volume, your validation needs prove that it actually does, and establish 
the extents to which your claim is true. Your validation plan needs to define how you would prove 
this, and establish these extents.

For the purpose of this exercise, assume that you have access to any clinical facility and patient 
cohorts you need, and that you have all the budget in the world. In your plan, touch on 
at least the following:

Your algorithm relies upon certain "ground truth" - how did you define your ground truth? How will 
you prove that your method of collecting the ground truth is robust and represents the population 
that you claim this algorithm is good for?
How do you define accuracy of your algorithm and how do you measure it with respect to real 
world population? Check out the calculator and report from HippoFit for some inspiration.
How do you define what data your algorithm can operate on?
There is no right answer here - think of these and other questions that would come up during 
validation of such algorithm. Thinking of such things early on will help you build better algorithms in the first place.


Please find `Section 3` code/data files [here](section3.tar.gz). Here is a tree diagram of the contents of 
section3 
```
section3
|-- README.txt
|-- data
|   |-- README.txt
|   `-- TestVolumes
|       `-- Study1
|           `-- 13_HCropVolume
|               |-- 1.dcm
|               |-- 10.dcm
|               |-- 11.dcm
|               |-- 12.dcm
|               |-- 13.dcm
|               |-- 14.dcm
|               |-- 15.dcm
|               |-- 16.dcm
|               |-- 17.dcm
|               |-- 18.dcm
|               |-- 19.dcm
|               |-- 2.dcm
|               |-- 20.dcm
|               |-- 21.dcm
|               |-- 22.dcm
|               |-- 23.dcm
|               |-- 24.dcm
|               |-- 25.dcm
|               |-- 26.dcm
|               |-- 27.dcm
|               |-- 28.dcm
|               |-- 29.dcm
|               |-- 3.dcm
|               |-- 30.dcm
|               |-- 31.dcm
|               |-- 32.dcm
|               |-- 4.dcm
|               |-- 5.dcm
|               |-- 6.dcm
|               |-- 7.dcm
|               |-- 8.dcm
|               `-- 9.dcm
|-- launch_OHIF.sh
|-- launch_orthanc.sh
|-- out
|   |-- DCM_report_OHIF_Viewer_screenshot.png
|   |-- PGBM-002_20230427_slice10.png
|   |-- PGBM-002_20230427_slice16.png
|   |-- PGBM-002_20230427_slice24.png
|   |-- README.md
|   |-- Validation_plan
|   |-- inference_dcm.py
|   `-- report.dcm
`-- src
    |-- <TEMPORARY\ PATH\ TO\ SAVE\ YOUR\ REPORT\ FILE>
    |-- assets
    |   `-- Roboto-Regular.ttf
    |-- data
    |   `-- TestVolumes
    |-- deploy_scripts
    |   |-- route_dicoms.lua
    |   |-- send_result.sh
    |   |-- send_volume.sh
    |   `-- start_listener.sh
    |-- environment.yml
    |-- inference
    |   |-- UNetInferenceAgent.py
    |   |-- __init__.py
    |   |-- __pycache__
    |   |   |-- UNetInferenceAgent.cpython-38.pyc
    |   |   `-- __init__.cpython-38.pyc
    |   `-- model.pth
    |-- inference_dcm.py
    |-- networks
    |   |-- RecursiveUNet.py
    |   |-- __init__.py
    |   `-- __pycache__
    |       |-- RecursiveUNet.cpython-38.pyc
    |       `-- __init__.cpython-38.pyc
    `-- utils
        |-- __pycache__
        |   |-- utils.cpython-38.pyc
        |   `-- volume_stats.cpython-38.pyc
        |-- utils.py
        `-- volume_stats.py

17 directories, 65 files

```

## Project Steps
This project has the following steps.


Acknowledgement: Udacity Nanodegree Team`
