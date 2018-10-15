%% LDA/QDA classifiers
%%feature dimentionality
featuresWhole=trainData;
featuresSubset=trainData(:,1:10:end);
labels=trainLabels;

% Using subset of features:

%Non-uniform (prior probabilities are empirically determined from labels)
classifierDiagLin=fitcdiscr(featuresSubset,labels,'DiscrimType','linear');

%Uniform classes (prior proba are considered equal)
classifierLinUnif=fitcdiscr(featuresSubset,labels,'DiscrimType','linear','prior','uniform');
classifierDiagLin=fitcdiscr(featuresSubset,labels,'DiscrimType','diaglinear','prior','uniform');
%classifierQuad=fitcdiscr(featuresSubset,labels,'DiscrimType','quadratic'); %covariance matrix SINGULAR 
classifierDiagQuad=fitcdiscr(featuresSubset,labels,'DiscrimType','diagquadratic','prior','uniform')

%Predictions 
yhatLin=predict(classifierDiagLin,featuresSubset);
yhatLinUnif=predict(classifierLinUnif,featuresSubset); %Uniform
yhatDiagLin=predict(classifierDiagLin,featuresSubset);
%yhatQuad=predict(classifierQuad,featuresSubset);
yhatDiagQuad=predict(classifierDiagQuad,featuresSubset);

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
% From previous section, we're working with : class error
% Divide dataset
set1 = trainData(1:2:end,1:10:end);
set2 = trainData(2:2:end,1:10:end);
label1 = trainLabels(1:2:end);
label2 = trainLabels(2:2:end);

ratio = 0.33;
% Comparison train and test errors
classifierDiagLin = fitcdiscr(set1,label1,'DiscrimType','diaglinear');
predictionDiagLin = predict(classifierDiagLin, set1);

trainingError = computeClassError(label1, predictionDiagLin, ratio);
testingError = computeClassError(label2, predictionDiagLin, ratio);
errorDiagLin = [trainingError, testingError]

%Same with linear, diagquadratic, quadratic
%Linear
classifierLin = fitcdiscr(set1,label1,'DiscrimType','linear');
predictionLin = predict(classifierLin, set1);

trainingError = computeClassError(label1, predictionLin, ratio);
testingError = computeClassError(label2, predictionLin, ratio);
errorLin = [trainingError, testingError]

%Diagquadratic
classifierDiagQuad = fitcdiscr(set1,label1,'DiscrimType','diagquadratic');
predictionDiagQuad = predict(classifierDiagQuad, set1);

trainingError = computeClassError(label1, predictionDiagQuad, ratio);
testingError = computeClassError(label2, predictionDiagQuad, ratio);
errorDiagQuad = [trainingError, testingError]

%Quadratic - isn't supposed to work
%classifierQuad = fitcdiscr(set1,trainLabels(1:2:end),'DiscrimType','quadratic');
%predictionQuad = predict(classifierQuad, set1);

%trainingError = computeClassError(label1, predictionQuad, ratio);
%testingError = computeClassError(label2, predictionQuad, ratio);
%errorQuad = [trainingError, testingError]

%- Would we still choose the same classifier ? -> linear has error = 0
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

