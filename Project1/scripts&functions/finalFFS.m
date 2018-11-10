%% Load datas sets

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testSet.mat');

%Normalize
[normData, mu, sigma] = zscore(trainData);
nsamples = size(testData,1);
nfeatures = size(testData,2);
meanTest = repmat(mu,nsamples,1); 
centeredTest = testData - meanTest;
stdTest = repmat(sigma,nsamples,1);
normTest = centeredTest ./ stdTest;

%% Build final model (entire dataset) for FFS

%Hyperparameters (from NCV):
model = 'quadratic';
Nsel = [10    11   110   128   131   133   174   187   202   204]; 

classifier = fitcdiscr(normData(:,Nsel), trainLabels,'discrimtype', model); 

%% Predictions on test set (testData.met)

yhat = predict(classifier, normTest(:, Nsel));
labelToCSV(yhat, 'final_quadratic_FFS.csv', '../submissions/');


