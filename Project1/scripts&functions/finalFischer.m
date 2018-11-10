%% Load datas sets

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testLabels.mat');

%% Build final model (entire dataset) for Fischer-based selection

%Hyperparameters (from NCV):

%% Predictions on test set (testData.met)
