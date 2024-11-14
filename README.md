# Bouldering Problem Difficulty Classification


## Contents:

This GitHub repository was put together by Philip Yao, Chris Kim, and Jessica Li. It contains source data, code, and output charts to analyze bouldering image data for the rock climbing wall at UVA.

### Section 1: Software and Platforms

- software: VS Code, Python Version 3.12.4

- add-on packages: numpy, Pillow, tqdm, ujson

- platform: mac
  

### Section 2: Documentation Hierarchy

- ds4002-project-2: Root Directory, Contains the following folders and files:

- ds4002-project-2/DATA: 
contains original dataset, cleaned dataset, and appendix

- ds4002-project-2/SCRIPTS: 
contains code used to clean, preprocess, and analyze image data

- ds4002-project-2/OUTPUT: 
contains plots and outputs

### Section 3: Instructions for Reproducability

- Install VS Code and Python Version 3.12.4

- Clone the git repository if working on google colab

- In the terminal, pip install requirements.txt; these are the necessary packages to perform the analysis

- Run these commands in the terminal in order to clean and preprocess the raw data:

```
cd PATH/TO/SCRIPTS
python binary_preprocess.py
python dataset_split.py
python image_preprocess.py
```

### References:

https://github.com/ColasGael/DeepClimb/tree/master/data/raw 

https://github.com/ColasGael/DeepClimb/blob/master/cs231n_final-report.pdf

