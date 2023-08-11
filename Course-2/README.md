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

## Section 2: Training a Segmentation CNN


















## Project Highlight


## Project Steps
This project has the following steps.


Acknowledgement: Udacity Nanodegree Team`
