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
errorDiagLin = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierDiagLin, ratio );

%Same with linear, diagquadratic, quadratic
%Linear
classifierLin = fitcdiscr(trainSet,trainLabels,'DiscrimType','linear');
errorLin = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierLin, ratio );

%Diagquadratic
classifierDiagQuad = fitcdiscr(trainSet,trainLabels,'DiscrimType','diagquadratic');
errorDiagQuad = trainTestError( trainSet, trainLabels, testSet, testLabels, classifierDiagQuad, ratio );

%Quadratic - isn't supposed to work

<<<<<<< HEAD
%- Would we still choose the same classifier ? -> linear has errors ~= 0.03
=======
%- Would we still choose the same classifier ?
>>>>>>> fa277867e75b71ba31a80b0ce2eeedc72ad886aa
%- Improvement on training error does not improve testing error as
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

N = length(trainData);
k = 10; %or we could set it up to 6 as we have ~600 samples, so size(subset)=100...?

%Partition comparison
cp_N = cvpartition(N, 'kfold', k);
cp_labels = cvpartition(trainLabels, 'kfold', k);

