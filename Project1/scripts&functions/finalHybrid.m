%% Load datas sets

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testSet.mat');

%Normalize
normData = zscore(trainData);
normTest = zscore(testData);

%% Build final model (entire dataset) for FFS with PCA

%Hyperparameters (from NCV):
model = 'quadratic';
Nsel = [10    11   110   128   131   133   174   187   202   204];


classifier = fitcdiscr(score(:,Nsel), trainLabels,'discrimtype', model); 

%% Predictions on test set (testData.met)

yhat = predict(classifier, normTest(:, Nsel));
labelToCSV(yhat, 'final_PCA_FFS.csv', '../submissions/');