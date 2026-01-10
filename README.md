# Advanced-ML-Project

Repository for the final project of the Advanced Machine Learning course (taught by Austin Stromme during the 1st Semester of the final year at ENSAE Paris).
We tried here to put the lottery ticket hypothesis, introduced by Frankle and Carbin in 2019, into practice by finding winning tickets in the PhysioNet Computing in Cardiology challenge 2017 dataset. Our work relies mainly on the article "LTH-ECG: Lottery Ticket Hypothesis-based Deep Learning Model Compression for Atrial Fibrillation Detection from Single Lead ECG On Wearable and Implantable Devices " (Sahu et al., 2022) and is motivated by the will of significantly reducing the computational size of an ECG classification model without compromising its accuracy.

## Contents

## Loading the data 

To get the data and unzip it first run in your shell :

```wget -N https://physionet.org/files/challenge-2017/1.0.0/training2017.zip```

```unzip training2017.zip```

To put the data into a folder named 'data', run :

```mkdir -p data```

```mv training2017 data/training2017```

To get the references run also : 

```wget -N https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv```

```mv REFERENCE-v3.csv data/REFERENCE-v3.csv```

You may now run the file 'build_datasets.py' to finish putting the data under the correct form !
