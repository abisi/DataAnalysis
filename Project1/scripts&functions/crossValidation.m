% Cross-validation of the final model
clear all
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Parameters
N = size(trainData,1); 
k = 5; 
%Hyperparameters
classifierType = ''; %TO FILL IN
Nsel = []; % TO FILL IN

cp_labels = cvpartition(trainLabels, 'kfold', k)
validationErrors = zeros(cp_labels.NumTestSets,1);

for i = 1:cp_labels.NumTestSets
    %choose subset
    trainId = cp_labels.training(i); 
    testId = cp_labels.test(i); 
    %train and predict
    classifier = fitcdiscr(trainData(trainId, Nsel), trainLabels(trainId, :), 'DiscrimType', classifierType);
    yhat = predict(classifier, trainData(testId, Nsel));
    %compute validation error
    validationErrors(i) = computeClassError(trainLabels(testId,:), yhat);    
end

meanValidationError = mean(validationErrors,1)
%Stability of performance : standard deviation of cross-validation error
stdValidationError = std(validationErrors)

boxplot(validationErrors, 'medianstyle', 'target'); hold on
ylabel('Validation error')
title('Boxplot of cross-validated final model')

%Statistical significance
[h, pvalue] = ttest(validationErrors, 0.5)

 

