# 📂 Dataset

This directory contains all dataset folders used for training and evaluation.

- **dataset_name** *(str)*  
  Name of the dataset folder in this directory  
  _Example_: `SMD`, `N_BaIoT`

- **subset_name** *(str)*  
  Name of the test split file  
  _Example_: `test`, `subset_A`

## Example Structure
```text
Dataset/
├── SMD/
│   ├── train.csv
│   └── test.csv
├── N_BaIoT/
│   ├── train.csv
│   └── subset_A.csv
└── Dataset.md
