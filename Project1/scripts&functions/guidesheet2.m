%% LDA/QDA classifiers
%feature dimentionality reduction
features=trainData(:,1:10:end);
labels=trainLabels;

%Establishing classification models:
classifierLin=fitcdiscr(features,labels,'DiscrimType','linear');
%Uniform classes (prior proba are considered equal)
classifierLinUnif=fitcdiscr(features,labels,'DiscrimType','linear','prior','uniform');
classifierDiagLin=fitcdiscr(features,labels,'DiscrimType','diaglinear','prior','uniform');
%classifierQuad=fitcdiscr(featuresSubset,labels,'DiscrimType','quadratic'); %covariance matrix SINGULAR 
classifierDiagQuad=fitcdiscr(features,labels,'DiscrimType','diagquadratic','prior','uniform')

%Predictions using models
yhatLin=predict(classifierLin,features);
yhatLinUnif=predict(classifierLinUnif,features); %Uniform
yhatDiagLin=predict(classifierDiagLin,features);
%yhatQuad=predict(classifierQuad,featuresSubset);
yhatDiagQuad=predict(classifierDiagQuad,features);

%Classification error/accuracy calculations for each classifier:
[errorLin, accurLin] = computeClassificationError(trainLabels, yhatLin);
[errorLinUnif, accurLinUnif] = computeClassificationError(trainLabels, yhatLinUnif);
%[errorQuad, accurQuad] = computeClassificationError(trainLabels, yhatLinQuad)
[errorDiagLin, accurDiagLin] = computeClassificationError(trainLabels, yhatDiagLin);
[errorDiagQuad, accurDiagQuad] = computeClassificationError(trainLabels, yhatDiagQuad);

% Class error calculations for each classifier:
[classErrorLin] = computeClassError(trainLabels, yhatLin, 0.5);
[classErrorLinUnif] = computeClassError(trainLabels, yhatLinUnif, 0.5);
[classErrorDiagLin] = computeClassError(trainLabels, yhatDiagLin, 0.5);
%[classErrorQuad] = computeClassError(trainLabels, yhatQuad, 0.5);
[classErrorDiagQuad] = computeClassError(trainLabels, yhatDiagQuad, 0.5);


%% Version 2.0
%If this part of the code is correct, we can remove all the part above.
%Small variations by comparing Class errors and classification errors >
%check together

clear all;
load('trainSet.mat');
load('trainLabels.mat');
labels=trainLabels;
features=trainData(:,1:10:end);

ratio=0.5;

classifierTypes=["linear","diaglinear","diagquadratic"]; %"quadratic" not used because singular problem 

%Compute classification error and class error for classifier: 
classificationErrors=[];
classErrors=[];

priorProba="uniform";
classificationErrors_priorProba=[];
classErrors_priorProba=[];

for i=1:length(classifierTypes)
   ypred=classifierPrediction(features, labels, classifierTypes(i));
  
   classificationError=computeClassificationError(labels, ypred);
   classificationErrors=[classificationErrors classificationError];
   
   classError = computeClassError(labels, ypred, ratio);
   classErrors = [classErrors, classError];
   
   %Case with prior probability = uniform
   ypred_priorProba=classifierPrediction(features, labels, classifierTypes(i), priorProba); %specifiy the prior proba to the function
   
   classificationError_priorProba=computeClassificationError(labels, ypred_priorProba);
   classificationErrors_priorProba=[classificationErrors_priorProba classificationError_priorProba];
   
   classError_priorProba = computeClassError(labels, ypred_priorProba, ratio);
   classErrors_priorProba = [classErrors_priorProba, classError_priorProba];
   
end

%Display results - Classification error
figure;
bar(classificationErrors);
set(gca,'XTickLabel',{'Linear','Diag Linear', 'Diag Quadratic'});
ylabel('Classification Error');
title('Classification error for different classifier methods');

%Display results - Class error
figure;
bar(classErrors);
set(gca,'XTickLabel',{'Linear','Diag Linear', 'Diag Quadratic'});
ylabel('Class Error');
title(['Class error for different classifier methods (ratio= ', num2str(ratio), ')' ]);


%Display results - Classification error, prior probability = uniform
figure;
bar(classificationErrors_priorProba);
set(gca,'XTickLabel',{'Linear','Diag Linear', 'Diag Quadratic'});
ylabel('Class Error');
title({'Classification error for different classifier methods'; 'Prior probability: uniform'});

%Display results - Class error, prior probability = uniform
figure;
bar(classErrors_priorProba);
set(gca,'XTickLabel',{'Linear','Diag Linear', 'Diag Quadratic'});
ylabel('Class Error');
title({['Class error for different classifier methods (ratio= ', num2str(ratio), ')' ]; 'Prior probability: uniform'});



%% Training and testing error
%clearing previous vars
clear all;
load('trainSet.mat');
load('trainLabels.mat');

% From previous section, we're working with : class error
% Divide dataset
trainSet = trainData(1:2:end,1:10:end);
testSet = trainData(2:2:end,1:10:end);
trainLabels = trainLabels(1:2:end);
testLabels = trainLabels(2:2:end);

ratio = 0.33;

% Comparison train and test errors
classifierDiagLin = fitcdiscr(trainSet,trainLabels,'DiscrimType','diaglinear');
errorDiagLin = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierDiagLin, ratio )

%Same with linear, diagquadratic, quadratic
%Linear
classifierLin = fitcdiscr(trainSet,trainLabels,'DiscrimType','linear');
errorLin = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierLin, ratio )

%Diagquadratic
classifierDiagQuad = fitcdiscr(trainSet,trainLabels,'DiscrimType','diagquadratic');
errorDiagQuad = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierDiagQuad, ratio )

%Quadratic - isn't supposed to work
%- Would we still choose the same classifier ? -> linear has errors ~= 0.03 
%- Yes : linear
classifierLin = fitcdiscr(trainSet,trainLabels,'DiscrimType','linear');
errorLinTest = trainTestError(testSet, testLabels, testSet, testLabels, classifierLin, ratio )
%- Improvement on training error does not improve testing error as...
%- Can't use quadratic : covariance matrix is SINGULAR i.e. not invertible

%Complexity - number of parameters
%Diaglinear:
%Linear:
%Diagquadratic:
%Number of training samples:
%- Are these classifiers robust ?


%% Again but revert set1 & set2 : training on set 2
%Notive the performance variability 



%% Cross validation for performance estimation

%Param.
ratio = 1/3;
N = size(trainData,1);
k = 10; %or we could set it up to 6 as we have ~600 samples, so size(subset)=100...?

%1. Partition comparison

cp_N = cvpartition(N, 'kfold', k);
cp_labels = cvpartition(trainLabels, 'kfold', k);
%Samples per class and per fold :
% cp_N: 1843-1844 per fold (per class)
% cp_labels: 269-270 per fold
% Would use cp_N because MORE DATA.


%2. k-fold cross validation
errorLinTest = zeros(cp_labels.NumTestSets,1);
for i = 1:cp_labels.NumTestSets
    trainId = cp_labels.training(i);
    testId = cp_labels.test(i);
    %train and predict
    classifierLin = fitcdiscr(trainData(trainId,:), trainLabels(trainId,:), 'DiscrimType', 'linear');
    yhatLin = predict(classifierLin, trainData(testId,:));
    %compute test error
    errorLinTest(i) = computeClassificationError(trainLabels(testId,:), yhatLin);    
end
meanError = mean(errorLinTest,1)
stdError = std(errorLinTest) %Stability of performance : standard deviation of cross-validation error


%3. Repeat with reparition(cp)
errorLinTest = zeros(cp_labels.NumTestSets,1);
for i = 1:cp_labels.NumTestSets
    repartition(cp_labels); %just rerandomizes, doesn't provide diff. performance
    trainId = cp_labels.training(i);
    testId = cp_labels.test(i);
    %train and predict
    classifierLin = fitcdiscr(trainData(trainId,:), trainLabels(trainId,:), 'DiscrimType', 'linear');
    yhatLin = predict(classifierLin, trainData(testId,:));
    %compute test error
    errorLinTest(i) = computeClassificationError(trainLabels(testId,:), yhatLin);    
end
meanError_2 = mean(errorLinTest,1)
%Stability of performance : standard deviation of cross-validation error
stdError_2 = std(errorLinTest)

% -No changes using repartition(cp_N). OK.
% -Why: repartition only rerandomize the partition -> should not affect
% result by much
% -Advantages of varying or similar classification performances : if
% similar, means preliminary model on cross-validation is consistent thus
% stable
% - No, model is made from entire training dataset -> cross validation only
% gives indication on the quality of the model

