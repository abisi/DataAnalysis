%% Nested cross-validation; feature ranking using fisher ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%outer/inner fold counts
kOuter=4;
kInner=5;

%observation count, upper threashold for nbFeatures hyperParam
nObservations=length(trainLabels);
maxN_features=300;

%Storage for training errors of models evaluated in inner fold cycles;
%models are differentiated based on how they are trained and nbFeatures.
%Each column of matrix corresponds to a new inner fold cycle; each row of
%matrix correponds to a different number of features.
DiagLin.InnerTrainingErrorStorage=zeros(maxN_features,kInner);
Lin.InnerTrainingErrorStorage=zeros(maxN_features,kInner);
DiagQuad.InnerTrainingErrorStorage=zeros(maxN_features,kInner);

%Storage for validation errors of models evaluated in inner fold cycles,
%same matrix dimensions correspondances as for InnerTrainingErrorStorage.
DiagLin.ValidationErrorStorage=zeros(maxN_features,kInner);
Lin.ValidationErrorStorage=zeros(maxN_features,kInner);
DiagQuad.ValidationErrorStorage=zeros(maxN_features,kInner);

%Storage of mean validation errors of models evaluated in inner fold
%cycles. Mean is taken across inner fold cycles, so we get avg validation
%error of a model using a specific number of features. Vector.
DiagLin.MeanValidationErrorStorage=zeros(maxN_features,1);
Lin.MeanValidationErrorStorage=zeros(maxN_features,1);
DiagQuad.MeanValidationErrorStorage=zeros(maxN_features,1);

%Sroage of mean training errors of models evaluated in inner fold cycles,
%same matrix dimensions correspondances as for MeanValidationErrorStorage.
DiagLin.MeanInnerTrainingErrorStorage=zeros(maxN_features,1);
Lin.MeanInnerTrainingErrorStorage=zeros(maxN_features,1);
DiagQuad.MeanInnerTrainingErrorStorage=zeros(maxN_features,1);

%storage of mean validation error for optimal model emergining from inner
%fold cycles
DiagLin.OptimalMeanValidationErrorStorage=zeros(1,kOuter);
Lin.OptimalMeanValidationErrorStorage=zeros(1,kOuter);
DiagQuad.OptimalMeanValidationErrorStorage=zeros(1,kOuter);

%storage of mean (inner)training error for optimal model emerging from
%inner fold cycles
DiagLin.OptimalMeanInnerTrainingErrorStorage=zeros(1,kOuter);
Lin.OptimalMeanInnerTrainingErrorStorage=zeros(1,kOuter);
DiagQuad.OptimalMeanInnerTrainingErrorStorage=zeros(1,kOuter);
        
%Storage of hyperParameters of optimal models emerging from inner loop
%cycles. Each bunch of inner loop cycles yiels an optimal model which is
%then tester against outer test section.
DiagLin.OptimalHyperParamStorage=zeros(1,kOuter);
Lin.OptimalHyperParamStorage=zeros(1,kOuter);
DiagQuad.OptimalHyperParamStorage=zeros(1,kOuter);

%Storage of optimal Model errors from outer fold testing 
%(trained with complete outerTraining section)
DiagLin.TestErrorStorage=zeros(1,kOuter);
Lin.TestErrorStorage=zeros(1,kOuter);
DiagQuad.TestErrorStorage=zeros(1,kOuter);

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
            
            %train classifiers using selected features, we also specify
            %priors
            diagLinearClassifier = fitcdiscr(innerTrainingSet(:,selectedFeatures), innerTrainingLabels, 'DiscrimType', 'diaglinear','Prior','uniform');
            linearClassifier = fitcdiscr(innerTrainingSet(:,selectedFeatures), innerTrainingLabels, 'DiscrimType', 'linear','Prior','uniform');
            diagQuadraticClassifier = fitcdiscr(innerTrainingSet(:,selectedFeatures), innerTrainingLabels, 'DiscrimType', 'diagquadratic','Prior','uniform');
            
            %use classifiers to predict classifications of
            %inner training section data
            dL_innerTrainingPrediction=predict(diagLinearClassifier, innerTrainingSet(:,selectedFeatures));
            L_innerTrainingPrediction=predict(linearClassifier, innerTrainingSet(:,selectedFeatures));
            dQ_innerTrainingPrediction=predict(diagQuadraticClassifier, innerTrainingSet(:,selectedFeatures));
            
            %compute classifier training error
            dL_trainingError= computeClassError(innerTrainingLabels,dL_innerTrainingPrediction);
            L_trainingError= computeClassError(innerTrainingLabels,L_innerTrainingPrediction);
            dQ_trainingError= computeClassError(innerTrainingLabels,dQ_innerTrainingPrediction);
            
            %use classifiers to predict classifications of validation
            %section data
            dL_validationPrediction=predict(diagLinearClassifier, validationSet(:,selectedFeatures));
            L_validationPrediction=predict(linearClassifier, validationSet(:,selectedFeatures));
            dQ_validationPrediction=predict(diagQuadraticClassifier, validationSet(:,selectedFeatures));
            
            %compute classifier validation error
            dL_validationError = computeClassError(validationLabels,dL_validationPrediction);
            L_validationError = computeClassError(validationLabels,L_validationPrediction);
            dQ_validationError = computeClassError(validationLabels,dQ_validationPrediction);
            
            %store training error of classifier trained on inner fold cycle t using q most
            %powerful features
            DiagLin.InnerTrainingErrorStorage(q,t)=dL_trainingError;
            Lin.InnerTrainingErrorStorage(q,t)=L_trainingError;
            DiagQuad.InnerTrainingErrorStorage(q,t)=dQ_trainingError;
            
            %store validation error of classifier trained on inner fold cycle t using q most
            %powerful features
            DiagLin.ValidationErrorStorage(q,t)=dL_validationError;
            Lin.ValidationErrorStorage(q,t)=L_validationError;
            DiagQuad.ValidationErrorStorage(q,t)=dQ_validationError;
        end
    end
    
    %compute training error mean across all inner fold cycles of linear
    %model using q most powerful features ;results in a vector of mean
    %validation errors where index is number of features used
    DiagLin.MeanInnerTrainingErrorStorage=mean(DiagLin.InnerTrainingErrorStorage,2); % we take row averages of storage matrix hence param 2
    Lin.MeanInnerTrainingErrorStorage=mean(Lin.InnerTrainingErrorStorage,2);
    DiagQuad.MeanInnerTrainingErrorStorage=mean(DiagQuad.InnerTrainingErrorStorage,2);
    
    %compute validation error mean across all inner fold cycles of
    %linear model using q most powerful features
    DiagLin.MeanValidationErrorStorage=mean(DiagLin.ValidationErrorStorage,2);
    Lin.MeanValidationErrorStorage=mean(Lin.ValidationErrorStorage,2);
    DiagQuad.MeanValidationErrorStorage=mean(DiagQuad.ValidationErrorStorage,2);
    
    %lookup classifier displaying minimum mean validation error and
    %store how many features it uses
    [dL_lowestMeanValidationError, dL_optimal_nFeatures]=min(DiagLin.MeanValidationErrorStorage);
    [L_lowestMeanValidationError, L_optimal_nFeatures]=min(Lin.MeanValidationErrorStorage);
    [dQ_lowestMeanValidationError, dQ_optimal_nFeatures]=min(DiagQuad.MeanValidationErrorStorage);
    
    %store mean training error (across inner fold cycles) of optimal model;
    % Vector with as many entries as outer folds
    DiagLin.OptimalMeanInnerTrainingErrorStorage(1,i)=DiagLin.MeanInnerTrainingErrorStorage(dL_optimal_nFeatures);
    Lin.OptimalMeanInnerTrainingErrorStorage(1,i)=Lin.MeanInnerTrainingErrorStorage(L_optimal_nFeatures);
    DiagQuad.OptimalMeanInnerTrainingErrorStorage(1,i)=DiagLin.MeanInnerTrainingErrorStorage(dQ_optimal_nFeatures);
    
    %store mean validation error of optimal model emerging from inner fold
    %cycles ; vector with as many entries as outer folds (kOuter)
    DiagLin.OptimalMeanValidationErrorStorage(1,i)=dL_lowestMeanValidationError;
    Lin.OptimalMeanValidationErrorStorage(1,i)=L_lowestMeanValidationError;
    DiagQuad.OptimalMeanValidationErrorStorage(1,i)=dQ_lowestMeanValidationError;
    
    %store optimal nbFeatures (hyperParam) of optimal model emerging from
    %inner fold cycles ; vector with as many entries as outer folds
    DiagLin.OptimalHyperParamStorage(1,i)=dL_optimal_nFeatures;
    Lin.OptimalHyperParamStorage(1,i)=L_optimal_nFeatures;
    DiagQuad.OptimalHyperParamStorage(1,i)=dQ_optimal_nFeatures;
    
    %Now we evaluate the optimal model emerging from inner folds in outer
    %fold cycle
    
    %rank features of outer fold cycle training section
    [orderedFeatureIndexes,orderedFeaturePowers] = rankfeat(outerTrainingSet, outerTrainingLabels, 'fisher');
    %select optimal number of most powerful features (optimal number is taken from optimal model emmerging from inner folds)
    dL_selectedFeatures=orderedFeatureIndexes(1:dL_optimal_nFeatures);
    L_selectedFeatures=orderedFeatureIndexes(1:L_optimal_nFeatures);
    dQ_selectedFeatures=orderedFeatureIndexes(1:dQ_optimal_nFeatures);
    
    %train optimal models using selected features; priors specification
    dL_optimalModel = fitcdiscr(outerTrainingSet(:,dL_selectedFeatures), outerTrainingLabels, 'DiscrimType', 'linear','Prior','uniform');
    L_optimalModel = fitcdiscr(outerTrainingSet(:,L_selectedFeatures), outerTrainingLabels, 'DiscrimType', 'linear','Prior','uniform');
    dQ_optimalModel = fitcdiscr(outerTrainingSet(:,dQ_selectedFeatures), outerTrainingLabels, 'DiscrimType', 'linear','Prior','uniform');
    
    %compute optimal model classification prediction for test section of
    %outer partition
    dL_optimalModelTestPrediction=predict(dL_optimalModel, testSet(:,dL_selectedFeatures));
    L_optimalModelTestPrediction=predict(L_optimalModel, testSet(:,L_selectedFeatures));
    dQ_optimalModelTestPrediction=predict(dQ_optimalModel, testSet(:,dQ_selectedFeatures));
    
    %compute optimal model's test error
    dL_testError = computeClassError(testLabels,dL_optimalModelTestPrediction);
    L_testError = computeClassError(testLabels,L_optimalModelTestPrediction);
    dQ_testError = computeClassError(testLabels,dQ_optimalModelTestPrediction);
    
    %store optimal model's test error (we get a new optimal model for next
    %outer fold cycle); vector of length equal to number of outer fold
    %cycles (kOuter)
    DiagLin.TestErrorStorage(1,i)=dL_testError;
    Lin.TestErrorStorage(1,i)=L_testError;
    DiagQuad.TestErrorStorage(1,i)=dQ_testError;
end