%% Load datas sets

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testLabels.mat');

%Normalize (or after PCA) ?
%[normData, mu, sigma] = zscore(trainData);
nsamples = size(testData,1);
nfeatures = size(testData,2);
meanTest = repmat(mu,nsamples,1); 
centeredTest = testData - meanTest;
stdTest = repmat(sigma,nsamples,1);
normTest = centeredTest ./ stdTest;

%Hyperparameters (from NCV):
Nsel = ...;
model = 'diaglinear';

%% Build final model (entire dataset) with PCA (unsupervised) - CV


%Simple cross-validation loop with fix classifier for now! (diaglin)
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
k=5;
nObservations=length(trainLabels);

trainErrorStorage=zeros(1,k);
testErrorStorage=zeros(1,k);

%Normalize data before applying PCA
%trainData=zscore(trainData);

[coeff, score, variance]=pca(trainData); %As we're working with transformed features, we have to do it before

%Normalize the transformed data (after PCA)
[score, mu, sigma]=zscore(score);

nsamples = size(testData,1);
nfeatures = size(testData,2);

meanTest = repmat(mu,nsamples,1); 
centeredTest = testData - meanTest;

stdTest = repmat(sigma,nsamples,1);
normTest = centeredTest ./ stdTest;

for t=1:k
    partition = cvpartition(nObservations, 'kfold', k); %les data transformés ont toujours 597 observations
    %Marker    
    trainMarker=partition.training(t);
    testMarker=partition.test(t);
    %Subsets
    trainSet=score(trainMarker,:); %new features
    trainingLabels=trainLabels(trainMarker, :); %vrais labels associés ne changent pas.     
    testSet=score(testMarker, :);
    testLabels=trainLabels(testMarker,:);
        
    selectedComponents=trainSet(:, 1:Nsel); %Components are classified by importance order.
    classifier = fitcdiscr(selectedComponents, trainingLabels, 'DiscrimType', 'diaglinear', 'prior', Priors); 
    
    trainPrediction=predict(classifier, trainSet(:, 1:Nsel));
    testPrediction=predict(classifier, testSet(:, 1:Nsel));
            
    trainError=computeClassError(trainPrediction, trainingLabels);
    testError=computeClassError(testPrediction, testLabels);
            
    trainErrorStorage(N,t)=trainError;
    testErrorStorage(N,t)=testError;
        
end

%Compute average errors > mean makes the mean on columns
meanTrainError=(mean(trainErrorStorage'))';

meanTestError=(mean(testErrorStorage'))'
std(testErrorStorage)

%% Predictions on test set (testData.met)

classifier = fitcdiscr(normData(:,Nsel), trainLabels, 'DiscrimType', 'diaglinear', 'prior', Priors); 
yhat = predict(classifier, normTest(:, Nsel));
labelToCSV(yhat, 'final_diaglinear_PCA.csv', '../submissions/');