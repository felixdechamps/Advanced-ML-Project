# Advanced-ML-Project

We are planning to work on the Lottery Ticket Hypothesis (LTH), introduced by Frankle and Carbin in the paper « The Lottery Ticket Hypothesis : Finding sparse, trainable neural networks » (2019), which was mentioned in your guidelines document. Precisely, the lottery ticket hypothesis states that randomly-initialized, dense neural networks contain subnetworks, called winning tickets, that when trained in isolation, reach test accuracy comparable to the original network trained in a similar number of iterations. In order to find such winning tickets, the authors use iterative pruning with the particularity that, at the end of each iterations, weights are rewound to their pre-trained values. 

Our aim is to apply this method to find winning tickets in ECG (Electrocardiogram) classification neural networks. We have indeed been inspired by the article « LTH-ECG: Lottery Ticket Hypothesis-based Deep Learning Model Compression for Atrial Fibrillation Detection from Single Lead ECG On Wearable and Implantable Devices » (Sahu et al., 2022), in which the authors try to find winning tickets using the PhysioNet Computing in Cardiology (CinC) Challenge 2017 dataset, comprising ECG recordings.

First, we would like to reproduce the finding of a winning ticket in the state-of-the art model as described in the latter article, by looking at the model’s accuracy with respect to the pruning level applied, using both retraining techniques fine tuning (iteratively pruning without rewinding weights nor learning rate) and Frankle and Carbin’s weight rewinding.
Then, based on the article « Comparing rewinding and fine-tuning in neural networks » (Renda et al., 2020), we also plan to implement the learning-rate rewinding method and compare the results with those obtained previously. If we have the time and succeed in finding winning tickets, we also intend to implement the late-resetting rewinding method introduced in the article « Stabilizing the Lottery Ticket Hypothesis » (Frankle et al., 2020), which is useful when it is difficult to find winning tickets. 

To summarize : Therefore, the first step would be to reproduce the main part of the paper (i.e. the comparison of fine tuning and LTH’s weight rewinding), and the next steps would consist in comparing with other re-initialisation techniques mentioned above.

In practice, we will therefore use the PhysioNet Computing in Cardiology challenge 2017 dataset together with  a deep neural network architecture consisting of “33 convolutional layers followed by a linear output layer into a softmax. The network accepts raw ECG data as input (sampled at 200 Hz, or 200 samples per second), and outputs a prediction of one out of 12 possible rhythm classes every 256 input samples” as described in the article "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network" by AY Hannun et al. (2019). We will implement the different iterative pruning methods based on the pruning heuristic most commonly used in the cited papers : magnitude pruning. We will then plot the accuracies in each case according to the parameter reduction factor, to see which is leading to the best results.

PhysioNet : https://physionet.org/content/challenge-2017/1.0.0/

# Loading the data 

To get the data and unzip it first run in your shell :

wget -N https://physionet.org/files/challenge-2017/1.0.0/training2017.zip

unzip training2017.zip

To put the data into a folder named 'data', run :
mkdir -p Advanced-ML-Project/data
mv training2017 Advanced-ML-Project/data/training2017

Finally, run : 

wget -N https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv
mv REFERENCE-v3.csv Advanced-ML-Project/data/REFERENCE-v3.csv

You may now run the file 'build_datasets.py' to finish putting the data under the correct form !
