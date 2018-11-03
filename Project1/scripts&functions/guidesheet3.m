% Guidesheet3: Model selection and nested cross-validation

clear all
close all
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%% Cross-validation for hyperparameter optimization

Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
k=4;
nObservations=length(trainLabels);
maxN_features=10;

trainingErrorStorage=zeros(maxN_features,k);
testErrorStorage=zeros(maxN_features,k);
optimalHyperParamStorage=0;

    for t=1:k
        partition = cvpartition(nObservations, 'kfold', k);
        trainingMarker=partition.training(t);
        testMarker=partition.test(t);
        trainingSet=trainData(trainingMarker, :);
        trainingLabels=trainLabels(trainingMarker, :);
        testSet=trainData(testMarker, :);
        testLabels=trainLabels(testMarker,:);
        [ftIndex,ftPower] = rankfeat(trainingSet, trainingLabels, 'fisher');
        for q=1:maxN_features
            selectedFeatures=ftIndex(1:q); % ftIndex is a list of feature indexs ordered from most powerful to least (fisher scoring)
            classifier = fitcdiscr(trainingSet(:,selectedFeatures), trainingLabels, 'DiscrimType', 'diaglinear','Prior', Priors);
            prediction=predict(classifier, testSet(:,selectedFeatures));
            trainingPrediction=predict(classifier, trainingSet(:,selectedFeatures));
            testError = computeClassError(testLabels,prediction,0.5);
            trainingError= computeClassError(trainingLabels,trainingPrediction,0.5);
            testErrorStorage(q,t)=testError;
            trainingErrorStorage(q,t)=trainingError;
        end
    end

%Average errors (+std)
 meanTestErrorStorage=(mean(testErrorStorage'))';
 meanTrainErrorStorage=(mean(trainingErrorStorage'))';

stdTrainErrors = (std(trainingErrorStorage'))'; % returns std dev of each column in a row vector
stdTestErrors = (std(testErrorStorage'))';

%Let's plot this
%Train
%figure
plot(1:maxN_features, meanTrainErrorStorage, 'LineWidth', 2, 'Color', 'r'); hold on
xlabel('Number of features');
ylabel('Mean training error'); 

%Test
plot(1:maxN_features, meanTestErrorStorage, 'LineWidth', 2, 'Color', 'b'); 
xlabel('Number of features');
ylabel('Mean testing error'); hold off

%% Nested cross-validation for performance estimation
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%priors struct
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
%outer/inner fold counts
kOuter=3;
kInner=4;
%observation count, max hyperParamValue
nObservations=length(trainLabels);
maxN_features=200;
%Storage for training errors of inner cross-validation
innerTrainingErrorStorage=zeros(maxN_features,kInner);
%Storage for validation errors of inner cross-validation
validationErrorStorage=zeros(maxN_features,kInner);
%Storage of mean validation errors of inner cross-validation
meanTestErrorStorage=zeros(maxN_features,1);
%Sroage of main training errors of inner cross-validation
meanInnerTrainingErrorStorage=zeros(maxN_features,1);
%Storage of hyperparams emerging from inner cross-validation
optimalHyperParamStorage=zeros(1,kOuter);
%Storage of errors from outer fold testing
testErrorStorage=zeros(1,kOuter);
%storage of mean validation error for optimal model emergining from inner
%cross validaiton
optimalMeanValidationErrorStorage=zeros(1,kOuter);
%storage of inner cross-validation training error for emerging models
optimalTrainingErrorStorage=zeros(1,kOuter);

%break data into 3 folds
partition_outer = cvpartition(nObservations, 'kfold', kOuter);

for i=1:kOuter
    outerTrainingMarker=partition_outer.training(i);
    testMarker=partition_outer.test(i);
    outerTrainingSet=trainData(outerTrainingMarker, :); %2 folds of original partition
    outerTrainingLabels=trainLabels(outerTrainingMarker, :);
    testSet=trainData(testMarker, :); % 1 fold of original partition
    testLabels=trainLabels(testMarker, :);
    nOuterTrainingSet=size(outerTrainingSet,1);
    %break trainingData from outer fold into a new 4 fold partition
    for t=1:kInner
        partition = cvpartition(nOuterTrainingSet, 'kfold', kInner);
        trainingMarker=partition.training(i);
        tesMarker=partition.test(i);
        trainingSet=outerTrainingSet(trainingMarker, :);
        trainingLabels=outerTrainingLabels(trainingMarker, :);
        validationSet=outerTrainingSet(tesMarker, :);
        validationLabels=outerTrainingLabels(tesMarker,:);
        [ftIndex,ftPower] = rankfeat(trainingSet, trainingLabels, 'fisher');
        for q=1:maxN_features
            selectedFeatures=ftIndex(1:q); % ftIndex is a list of feature indexs ordered from most powerful to least (fisher scoring)
            classifier = fitcdiscr(trainingSet(:,selectedFeatures), trainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
            prediction=predict(classifier, validationSet(:,selectedFeatures));
            trainingPrediction=predict(classifier, trainingSet(:,selectedFeatures));
            validationError = computeClassError(validationLabels,prediction,0.5);
            trainingError= computeClassError(trainingLabels,trainingPrediction,0.5);
            validationErrorStorage(q,t)=validationError;
            innerTrainingErrorStorage(q,t)=trainingError;
        end
    end
    meanTestErrorStorage=(mean(validationErrorStorage'))';
    meanInnerTrainingErrorStorage=(mean(innerTrainingErrorStorage'))';
    [lowestMeanValidationError optimal_nFeatures]=min(meanTestErrorStorage);
    optimalMeanValidationErrorStorage(1,i)=lowestMeanValidationError;
    optimalTrainingErrorStorage(1,i)=meanInnerTrainingErrorStorage(optimal_nFeatures);
    optimalHyperParamStorage(1,i)=optimal_nFeatures;
    [ftIndex,ftPower] = rankfeat(outerTrainingSet, outerTrainingLabels, 'fisher');
    selectedFeatures=ftIndex(1:optimal_nFeatures);
    optimalModel = fitcdiscr(outerTrainingSet(:,selectedFeatures), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    prediction=predict(optimalModel, testSet(:,selectedFeatures));
    testError = computeClassError(testLabels,prediction, 0.5); 
    testErrorStorage(1,i)=testError;
end
%% Subsequent questions
% How does hyperparameter differ across folds?
optimalHyperParamStorage
%-> number of optimal features varies a lot across folds!

% Boxplots of mean training/validation/test errors for optimal models
A=optimalTrainingErrorStorage;
B=optimalMeanValidationErrorStorage;
C=testErrorStorage;

ToPlot=[A;B;C];
groupingMatrix= [ ones(size(A)); 2 * ones(size(B)); 3 * ones(size(C))];
figure; hold on
boxplot(ToPlot,groupingMatrix);
set(gca,'XTickLabel',{'A','B','C'});
%Validation error is much more unstable than both training and test errors.

%Make model type the hyperparameter to determine
%-> No point implementing until we are sure our current code is correct.
%-> Potential issue : optimal hyperParameter variability emerging from
%inner cross-validation.