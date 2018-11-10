%% Load datas sets

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testLabels.mat');

%% Build final model (entire dataset) with PCA (unsupervised)

%Hyperparameters (from NCV):

%% Predictions on test set (testData.met)
