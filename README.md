# Bouldering Problem Difficulty Classification


## Contents:

This GitHub repository was put together by Philip Yao, Chris Kim, and Jessica Li. It contains source data, code, and output charts to analyze bouldering image data for the rock climbing moonboard.

### Section 1: Software and Platforms

- software: VS Code, Anaconda, Python Version 3.12.4

- add-on packages: numpy, Pillow, tqdm, ujson, torch

- platform: mac
  

### Section 2: Documentation Hierarchy

- ds4002-project-2: Root Directory, Contains the following folders and files:

- ds4002-project-2/DATA: 
contains original raw dataset, binary files, image files, and appendix

    - Note: binary and image folders/files will not show until after the preprocessing scripts are run, as shown in section 3.

- ds4002-project-2/SCRIPTS: 
contains code used to clean, preprocess, and analyze image data

- ds4002-project-2/OUTPUT: 
contains plots and other image outputs

### Section 3: Instructions for Reproducability

- Install VS Code, Anaconda, and Python Version 3.12.4

- Clone the git repository

- Create a conda env

- In the terminal, pip install requirements.txt; these are the necessary packages to perform the analysis.

```
cd PATH/TO/SCRIPTS
pip install -r requirements.txt
```

- Run these commands in the terminal in order to clean and preprocess the raw data:

```
python binary_preprocess.py
python dataset_split.py
python image_preprocess.py
```

- Run through code.ipynb if you are interested in looking at the processed image dataset.

- Run these commands in the terminal in order to define the CNN model functions:

```
python CNN_models.py
```

### References:

https://github.com/ColasGael/DeepClimb/tree/master/data/raw 


