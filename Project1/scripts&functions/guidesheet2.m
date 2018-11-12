% Guidesheet2: Linear/Quadratic Discriminant Analysis Classifiers

clear all
close all
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
load('../data/testSet.mat');

%% LDA/QDA Classifiers

%Initialize:
labels=trainLabels;
classificationErrors = []; 
scores = [];
%Feature subset
features=trainData(:, 1:50:end);
%LDA/QDA
classifierTypes = ["linear", "diaglinear", "diagquadratic"]; %"quadratic" not used because singular problem  

for i=1:length(classifierTypes)
    
   classifier = fitcdiscr(features, labels, 'DiscrimType', char(classifierTypes(i)));
   yhat = predict(classifier, features);
   [classificationErrors(i), scores(i)] = computeClassificationError(labels, yhat);
    
end

%Display results - Classification error
classificationErrors, scores

%Based on classification error, we'd use LINEAR classifier because error is 0.

%% Prior probabilities - UNIFORM

classificationErrors = []; 
classErrors = [];
classificationScores = [];
classScores = [];

for i=1:length(classifierTypes)
    
    %'uniform' sets all class probabiblities equal (although not true here)
   classifier = fitcdiscr(features, labels, 'DiscrimType', char(classifierTypes(i)), 'Prior', 'uniform');
   yhat = predict(classifier, features);
   [classificationErrors(i), classificationScores(i)] = computeClassificationError(labels, yhat);
   [classErrors(i), classScores(i)] = computeClassError(labels, yhat);
    
end

%Display results - Classification error
classificationErrors, classificationScores
classErrors, classScores
%With UNIFORM : gives absolutely the same thing which is normal since
%uniform sets to equal probabilities.

%% Prior probabilities - EMPIRICAL (default)

classificationErrors = []; 
classErrors = [];
classificationScores = [];
classScores = [];

for i=1:length(classifierTypes)
    
    %'uniform' sets all class probabiblities equal (although not true here)
   classifier = fitcdiscr(features, labels, 'DiscrimType', char(classifierTypes(i)), 'Prior', 'empirical');
   yhat = predict(classifier, features);
   [classificationErrors(i), classificationScores(i)] = computeClassificationError(labels, yhat);
   [classErrors(i), classScores(i)] = computeClassError(labels, yhat);
    
end

%Display results - Classification error
classificationErrors, classificationScores
classErrors, classScores
%With EMPIRICAL : gives absolutely the same thing which is normal since
%empirical sets finds class probabilities. Better performance with
%empirical because does not artificially set equal probabilities (not the
%case with our data).


%% Summary: Assuming that computeClassError already has the ratio 2/3-1/z3. 
% With uniform prior, classifier assumes equal class probabilities, which is not the case. 
% Thus, whether we compute classification error or class error does not matter because 
% we have already assumed equal classes thus both functions return the same
% results.
% Without uniform prior, i.e. empirical, classifier found with calculated
% class probabilities (by fitcdiscr). Scores are higher since the real
% class distributions is taken into account.
% Thus, we will use CLASS ERROR from now on, ASSUMING that the class distributions will remain the same.

%% Training and testing error

set1 = trainData(1:2:end,1:5:end); %set1
set2 = trainData(2:2:end,1:5:end);  %set2

reducedSet1 = trainData(1:2:end,1:30:end); %because too long...?
reducedSet2 = trainData(2:2:end,1:30:end);

labels1 = trainLabels(1:2:end);
labels2 = trainLabels(2:2:end);

% Comparison train and test errors
classifierDiagLin = fitcdiscr(set1, labels1, 'DiscrimType','diaglinear');
yhatDiagLin = predict(classifierDiagLin, set1);
trainingError = computeClassError(labels1, yhatDiagLin)
testingError = computeClassError(labels2, yhatDiagLin)

% Training and testing error are similar, yet slightly higher on set2
% testing set.

%% Again with different classifier types

classifierTypes = ["linear", "diagquadratic"]; % quadratic not used because SINGULAR matrix
trainingErrors = [];
testingErrors = [];

% Comparison train and test errors
for i=1:length(classifierTypes)
    classifier = fitcdiscr(set1, labels1, 'DiscrimType', char(classifierTypes(i)));
    yhatDiagLin = predict(classifier, set1);
    trainingErrors(i) = computeClassError(labels1, yhatDiagLin);
    testingErrors(i) = computeClassError(labels2, yhatDiagLin);
end

%Display
trainingErrors, testingErrors

% -We would still choose LINEAR because error is 0.
% -In addition, improvement on training error not reflected in testing error
% because UNSEEN data set ? 
% -Robustness of classifiers and number of parameters:


%% Training and testing error, but TRAIN on set2

% Comparison train and test errors
classifierDiagLin = fitcdiscr(set2, labels2, 'DiscrimType','diaglinear');
yhatDiagLin = predict(classifierDiagLin, set1);
trainingError = computeClassError(labels1, yhatDiagLin)
testingError = computeClassError(labels2, yhatDiagLin)

%Different performance: training and testing errors both greater than when
%trained on set1. 

%% Modify prior to UNIFORM

% Comparison train and test errors
classifierDiagLin = fitcdiscr(set2, labels2, 'DiscrimType','diaglinear', 'Prior', 'uniform');
yhatDiagLin = predict(classifierDiagLin, set1);
trainingError = computeClassError(labels1, yhatDiagLin)
testingError = computeClassError(labels2, yhatDiagLin)

%Different performance: training and testing errors EVEN greater with UNIFORM.

%% Kaggle 
%choose classifier and subsets
%make prediction on testData
filename = 'guidesheet2_diaglin_empirical_trainonset1.csv';
folder = pwd;
labelToCSV(yhat, filename, folder);

%Bad scores because seen data
%% Cross validation for performance estimation

%Param.
N = size(trainData,1); %597
k = 10; %or we could set it up to 6 as we have ~600 samples, so size(subset)=100...?

%% 1. Partition comparison

cp_N = cvpartition(N, 'kfold', k)
cp_labels = cvpartition(trainLabels, 'kfold', k)

%Samples per class and per fold :
% cp_N: 538-637 per fold
% cp_labels: same
% Would use cp_labels because takes into account class distributino ! 


%% 2. k-fold cross validation
errorLinTest = zeros(cp_labels.NumTestSets,1);
for i = 1:cp_labels.NumTestSets
    trainId = cp_labels.training(i); %marks training samples with 1
    testId = cp_labels.test(i); % marks testing samples with 1
    %train and predict
    classifierLin = fitcdiscr(trainData(trainId,:), trainLabels(trainId,:), 'DiscrimType', 'linear');
    yhatLin = predict(classifierLin, trainData(testId,:));
    %compute test error
    errorLinTest(i) = computeClassificationError(trainLabels(testId,:), yhatLin);    
end
meanError = mean(errorLinTest,1)
stdError = std(errorLinTest) %Stability of performance : standard deviation of cross-validation error


%% 3. Repeat with repartition(cp)
errorLinTest2 = zeros(cp_labels.NumTestSets,1);
for i = 1:cp_labels.NumTestSets
    cp_labels=repartition(cp_labels); %just rerandomizes, doesn't provide diff. performance
    trainId = cp_labels.training(i);
    testId = cp_labels.test(i);
    %train and predict
    classifierLin = fitcdiscr(trainData(trainId,:), trainLabels(trainId,:), 'DiscrimType', 'linear');
    yhatLin = predict(classifierLin, trainData(testId,:));
    %compute test error
    errorLinTest2(i) = computeClassificationError(trainLabels(testId,:), yhatLin);    
end
meanError_2 = mean(errorLinTest2,1)
%Stability of performance : standard deviation of cross-validation error
stdError_2 = std(errorLinTest2)

% -No big changes using repartition(cp_N): 0.03. OK.
% -Why: repartition only rerandomize the partition -> should not affect
% result by much
% -Advantages of varying or similar classification performances : if
% similar, means preliminary model on cross-validation is consistent thus
% stable
% - No, model is made from entire training dataset -> cross validation only
% gives indication on the quality of the model

