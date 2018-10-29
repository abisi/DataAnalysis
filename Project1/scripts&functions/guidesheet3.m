%% Guidesheet3: Model selection and nested cross-validation

%%Cross-validation for hyperparameter optimization

%Fisher-based feature selection
k = 10; %number of folds
maxFt = 20; %max number of features
cp_labels = cvpartition(trainLabels, 'kfold', k);%partition
trainErrors = zeros(k, maxFt); %initialization
testErrors = zeros(k, maxFt);

for ft = 1:maxFt %incrementing number of features
    for i=1:cp_labels.NumTestSets
        trainId = cp_labels.training(i); %markers for training
        testId = cp_labels.test(i); %markers for testing
        %Pick ft-th best features
    	[orderedInd, orderedPower] = rankfeat(trainData(trainId,:), trainLabels(trainId,:), 'fisher');
        bestInd = orderedInd(1:ft); %selection of best n features
        %Build model and predict
        classifierDiagLin = fitcdiscr(trainData(trainId, bestInd), trainLabels(trainId,:), 'DiscrimType', 'diaglinear');
        yhatTrain = predict(classifierDiagLin, trainData(trainId, bestInd));
        yhatTest = predict(classifierDiagLin, trainData(testId, bestInd));
        %Compute both errors
        errTrain = computeClassError(trainLabels(trainId,:), yhatTrain, 0.5);
        errTest = computeClassError(trainLabels(testId,:), yhatTest, 0.5);
        %Store 
        trainErrors(i, ft) = errTrain; % Each column respresents a different model (different amount of features used)
        testErrors(i, ft) = errTest; % Each row of selected column contains different fold respective errors.
    end
end

%Average errors (+std)
meanTrainErrors = mean(trainErrors) %mean of a matrix returns row vector containing mean value of each column
meanTestErrors = mean(testErrors)

stdTrainErrors = std(trainErrors); % returns std dev of each column in a row vector
stdTestErrors = std(testErrors);

%Let's plot this
%Train
figure
plot(1:maxFt, meanTrainErrors, 'LineWidth', 2, 'Color', 'r'); hold on
xlabel('Number of features');
ylabel('Mean training error'); 

%Test
plot(1:maxFt, meanTestErrors, 'LineWidth', 2, 'Color', 'b'); 
xlabel('Number of features');
ylabel('Mean testing error'); hold off

%% Nested cross-validation for performance estimation
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
kOuter=3;
kInner=4;
nObservations=length(trainLabels);
maxN_features=100;
validationErrorStorage=zeros(maxN_features,kInner);
meanValidationErrorStorage=zeros(maxN_features,1);
optimalHyperParamStorage=zeros(1,kOuter);
testErrorStorage=zeros(1,kOuter);

partition_outer = cvpartition(nObservations, 'kfold', kOuter);

for i=1:kOuter
    outerTrainingMarker=partition_outer.training(i);
    testMarker=partition_outer.test(i);
    outerTrainingSet=trainData(outerTrainingMarker, :);
    outerTrainingLabels=trainLabels(outerTrainingMarker, :);
    testSet=trainData(testMarker, :);
    testLabels=trainLabels(testMarker, :);
    nOuterTrainingSet=size(outerTrainingSet,1);
    for t=1:kInner
        partition_inner = cvpartition(nOuterTrainingSet, 'kfold', kInner);
        innerTrainingMarker=partition_inner.training(i);
        validationMarker=partition_inner.test(i);
        innerTrainingSet=outerTrainingSet(innerTrainingMarker, :);
        innerTrainingLabels=outerTrainingLabels(innerTrainingMarker, :);
        validationSet=outerTrainingSet(validationMarker, :);
        validationLabels=outerTrainingLabels(validationMarker,:);
        [ftIndex,ftPower] = rankfeat(innerTrainingSet, innerTrainingLabels, 'fisher');
        for q=1:maxN_features
            selectedFeatures=ftIndex(1:q); % ftIndex is a list of feature indexs ordered from most powerful to least (fisher scoring)
            classifier = fitcdiscr(innerTrainingSet(:,selectedFeatures), innerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
            prediction=predict(classifier, validationSet(:,selectedFeatures));
            validationError = computeClassError(validationLabels,prediction,0.5);
            validationErrorStorage(q,t)=validationError;
        end
    end
    meanValidationErrorStorage=(mean(validationErrorStorage'))';
    [lowestMeanValidationError optimal_nFeatures]=min(meanValidationErrorStorage);
    optimalHyperParamStorage(1,i)=optimal_nFeatures;
    [ftIndex,ftPower] = rankfeat(outerTrainingSet, outerTrainingLabels, 'fisher');
    selectedFeatures=ftIndex(1:optimal_nFeatures);
    optimalModel = fitcdiscr(outerTrainingSet(:,selectedFeatures), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    prediction=predict(optimalModel, testSet(:,selectedFeatures));
    testError = computeClassError(testLabels,prediction, 0.5); 
    testErrorStorage(1,i)=testError;
end