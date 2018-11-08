%% Nested cross-validation; feature ranking using fisher ; hyperParams: nbFeatures
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%priors struct:  we need to specify priors when fitting models to data
%where class frequencies are unknown
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

%outer/inner fold counts
kOuter=3;
kInner=4;

%observation count, upper threashold for nbFeatures hyperParam
nObservations=length(trainLabels);
maxN_features=200;

%Storage for training errors of models evaluated in inner fold cycles;
%models are differentiated based on how they are trained and nbFeatures.
%Each column of matrix corresponds to a new inner fold cycle; each row of
%matrix correponds to a different number of features.
innerTrainingErrorStorage=zeros(maxN_features,kInner);

%Storage for validation errors of models evaluated in inner fold cycles,
%same matrix dimensions correspondances as for innerTrainingErrorStorage.
validationErrorStorage=zeros(maxN_features,kInner);

%Storage of mean validation errors of models evaluated in inner fold
%cycles. Mean is taken across inner fold cycles, so we get avg validation
%error of a model using a specific number of features. Vector.
meanValidationErrorStorage=zeros(maxN_features,1);

%Sroage of main training errors of models evaluated in inner fold cycles,
%same matrix dimensions correspondances as for meanValidationErrorStorage.
meanInnerTrainingErrorStorage=zeros(maxN_features,1);

%storage of mean validation error for optimal model emergining from inner
%fold cycles
optimalMeanValidationErrorStorage=zeros(1,kOuter);

%storage of mean (inner)training error for optimal model emerging from
%inner fold cycles
optimalMeanInnerTrainingErrorStorage=zeros(1,kOuter);
        
%Storage of hyperParameters of optimal models emerging from inner loop
%cycles. Each bunch of inner loop cycles yiels an optimal model which is
%then tester against outer test section.
optimalHyperParamStorage=zeros(1,kOuter);

%Storage of optimal Model errors from outer fold testing 
%(trained with complete outerTraining section)
testErrorStorage=zeros(1,kOuter);



%break total data into 3 outer folds
partition_outer = cvpartition(nObservations, 'kfold', kOuter);

%for each outer fold cycle
for i=1:kOuter
    %get training section indexes
    outerTrainingMarker=partition_outer.training(i);
    %get test section indexes
    testMarker=partition_outer.test(i);
    
    %load training section data
    outerTrainingSet=trainData(outerTrainingMarker, :); %2 folds of original partition
    %load training section labels
    outerTrainingLabels=trainLabels(outerTrainingMarker, :);
    
    %load test section data
    testSet=trainData(testMarker, :); % 1 fold of original partition
    %load test section labels
    testLabels=trainLabels(testMarker, :);
    
    %size of training section (in terms of nObservations)
    nOuterTrainingSet=size(outerTrainingSet,1);
    
    %break outer training section into a new 4 fold partition
    partition_inner = cvpartition(nOuterTrainingSet, 'kfold', kInner);
    
    %for each inner fold cycle
    for t=1:kInner
        %get inner training section indexes
        innerTrainingMarker=partition_inner.training(t);
        %get validation section indexes
        validationMarker=partition_inner.test(t);
        
        %load inner training section data
        innerTrainingSet=outerTrainingSet(innerTrainingMarker, :);
        %load inner training section labels
        innerTrainingLabels=outerTrainingLabels(innerTrainingMarker, :);
        
        %load validation section indexes
        validationSet=outerTrainingSet(validationMarker, :);
        %load validation section labels
        validationLabels=outerTrainingLabels(validationMarker,:);
        
        %rank inner training section features using fisher scoring
        [orderedFeatureIndexes,orderedFeaturePowers] = rankfeat(innerTrainingSet, innerTrainingLabels, 'fisher'); %rankFeat 
        
        %iterating over possible nbFeature hyperParam values
        for q=1:maxN_features
            %we select q most powerful features (based on fisher scoring)
            selectedFeatures=orderedFeatureIndexes(1:q);
            
            %train diagonal linear classifier using selected features, we
            %also specify priors
            diagLinearClassifier = fitcdiscr(innerTrainingSet(:,selectedFeatures), innerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
            
            %use diagonal linear classifier to predict classifications of
            %inner training section data
            dL_innerTrainingPrediction=predict(diagLinearClassifier, innerTrainingSet(:,selectedFeatures));
            %compute diagonal linear classifier training error
            dL_trainingError= computeClassError(innerTrainingLabels,dL_innerTrainingPrediction);
            
            %use diagonal linear classifier to predict classifications of
            %validation section data
            dL_validationPrediction=predict(diagLinearClassifier, validationSet(:,selectedFeatures));
            %compute diagonal linear classifier validation error
            validationError = computeClassError(validationLabels,dL_validationPrediction);
            
            %store training error of classifier trained on inner fold cycle t using q most
            %powerful features
            innerTrainingErrorStorage(q,t)=dL_trainingError;
            
            %store validation error of classifier trained on inner fold cycle t using q most
            %powerful features
            validationErrorStorage(q,t)=validationError;
        end
    end
    %compute training error mean across all inner fold cycles of diagonal
    %linear model using q most powerful features ;results in a vector of
    %mean validation errors where index is number of features used
    meanInnerTrainingErrorStorage=mean(innerTrainingErrorStorage,2); % we take row averages of storage matrix hence param 2
    %compute validation error mean across all inner fold cycles of diagonal
    %linear model using q most powerful features
    meanValidationErrorStorage=mean(validationErrorStorage,2);
    
    %lookup dL classifier displaying minimum mean validation error and
    %store how many features it uses
    [lowestMeanValidationError, optimal_nFeatures]=min(meanValidationErrorStorage);
    
    %store mean training error (across inner fold cycles) of optimal model;
    % Vector with as many entries as outer folds
    optimalMeanInnerTrainingErrorStorage(1,i)=meanInnerTrainingErrorStorage(optimal_nFeatures);
    
    %store mean validation error of optimal model emerging from inner fold
    %cycles ; vector with as many entries as outer folds (kOuter)
    optimalMeanValidationErrorStorage(1,i)=lowestMeanValidationError;
    
    %store optimal nbFeatures (hyperParam) of optimal model emerging from
    %inner fold cycles ; vector with as many entries as outer folds
    optimalHyperParamStorage(1,i)=optimal_nFeatures;
    
    %Now we evaluate the optimal model emerging from inner folds in outer
    %fold cycle
    
    %rank features of outer fold cycle training section
    [orderedFeatureIndexes,orderedFeaturePowers] = rankfeat(outerTrainingSet, outerTrainingLabels, 'fisher');
    %select optimal number of most powerful features (optimal number is taken from optimal model emmerging from inner folds)
    selectedFeatures=orderedFeatureIndexes(1:optimal_nFeatures);
    
    %train optimal model using selected features; priors specification
    optimalModel = fitcdiscr(outerTrainingSet(:,selectedFeatures), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    %compute optimal model classification prediction for test section of
    %outer partition
    optimalModelTestPrediction=predict(optimalModel, testSet(:,selectedFeatures));
    
    %compute optimal model's test error
    testError = computeClassError(testLabels,optimalModelTestPrediction);
    
    %store optimal model's test error (we get a new optimal model for next
    %outer fold cycle); vector of length equal to number of outer fold
    %cycles (kOuter)
    testErrorStorage(1,i)=testError;
end