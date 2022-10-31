# CS-433 Project 1
## Authors (team: mapodoufu)
- Zehao Chen
- Haolong Li
- Xuehan Tang

## Introduction
This repository contains the code for  [Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) of Machine Learning 2022 fall at EPFL.

The aim of this project is to recreate the process of “discovering” the Higgs particle, through the analysis of decay signature of protons collisions.

By introducing machine learning methods, we try to predict whether a collision was **signal** (a Higgs boson) or **background** (something else).

## File structure
```
│  acc.png
│  implementations.py
│  log_res.txt
│  README.md
│  report.pdf
│  run.py
│  shell_top.py
│
├─data
│      submission.csv
│      test.csv
│      train.csv
│
└─results
```
- acc.png: the graph which shows accuracy with iterations on the validation dataset.
- implementations.py: the file contains all implementations of ML techniques required for the project
- log_res.txt: the log of multi-process programming, aiming to find the best hyperparameters using grid search
- report.pdf: the report of project 1 built by latex.
- run.py: main script which train and test the model, then predict labels to generate submission file
- shell_top.py: python script to run multi-process programming, performing grid searching.
- data/ : data folder
- results/ : the folder to place submission results
## Reproduce our results

### Prepare the data
Download raw data files from the official [link]([link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files?unique_download_uri=172694&challenge_id=66))(https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files?unique_download_uri=172694&challenge_id=66), unzip them and put them under data/ folder.

### Run the code
From the root folder of the project

```shell
python run.py
```

Then the result will be under the results folder.
