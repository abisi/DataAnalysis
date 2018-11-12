%% Nested cross-validation; feature ranking using fisher ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

k=5;

nObservations=length(trainLabels);
maxN_features=300;

DiagLin.ValidationErrorStorage=zeros(maxN_features,k);
Lin.ValidationErrorStorage=zeros(maxN_features,k);
DiagQuad.ValidationErrorStorage=zeros(maxN_features,k);

DiagLin.MeanValidationErrorStorage=zeros(maxN_features,1);
Lin.MeanValidationErrorStorage=zeros(maxN_features,1);
DiagQuad.MeanValidationErrorStorage=zeros(maxN_features,1);

DiagLin.OptimalMeanValidationErrorStorage=0;
Lin.OptimalMeanValidationErrorStorage=0;
DiagQuad.OptimalMeanValidationErrorStorage=0;
        
DiagLin.OptimalHyperParamStorage=0;
Lin.OptimalHyperParamStorage=0;
DiagQuad.OptimalHyperParamStorage=0;

partition = cvpartition(nObservations, 'kfold', k);

    %for each inner fold cycle
    for t=1:k
        trainingMarker=partition.training(t);
        validationMarker=partition.test(t);
        
        trainingSet=trainData(trainingMarker, :);
        trainingLabels=trainLabels(trainingMarker, :);
        
        validationSet=trainData(validationMarker, :);
        validationLabels=trainLabels(validationMarker,:);
        
        [orderedFeatureIndexes,orderedFeaturePowers] = rankfeat(trainingSet, trainingLabels, 'fisher'); %rankFeat 
        
        for q=1:maxN_features
            selectedFeatures=orderedFeatureIndexes(1:q);
            
            diagLinearClassifier = fitcdiscr(trainingSet(:,selectedFeatures), trainingLabels, 'DiscrimType', 'diaglinear','Prior','uniform');
            linearClassifier = fitcdiscr(trainingSet(:,selectedFeatures), trainingLabels, 'DiscrimType', 'linear','Prior','uniform');
            diagQuadraticClassifier = fitcdiscr(trainingSet(:,selectedFeatures), trainingLabels, 'DiscrimType', 'diagquadratic','Prior','uniform');
            
            dL_validationPrediction=predict(diagLinearClassifier, validationSet(:,selectedFeatures));
            L_validationPrediction=predict(linearClassifier, validationSet(:,selectedFeatures));
            dQ_validationPrediction=predict(diagQuadraticClassifier, validationSet(:,selectedFeatures));
            
            dL_validationError = computeClassError(validationLabels,dL_validationPrediction);
            L_validationError = computeClassError(validationLabels,L_validationPrediction);
            dQ_validationError = computeClassError(validationLabels,dQ_validationPrediction);
            
            DiagLin.ValidationErrorStorage(q,t)=dL_validationError;
            Lin.ValidationErrorStorage(q,t)=L_validationError;
            DiagQuad.ValidationErrorStorage(q,t)=dQ_validationError;
        end
    end
        
    DiagLin.MeanValidationErrorStorage=mean(DiagLin.ValidationErrorStorage,2);
    Lin.MeanValidationErrorStorage=mean(Lin.ValidationErrorStorage,2);
    DiagQuad.MeanValidationErrorStorage=mean(DiagQuad.ValidationErrorStorage,2);
    
    [dL_lowestMeanValidationError, dL_optimal_nFeatures]=min(DiagLin.MeanValidationErrorStorage);
    [L_lowestMeanValidationError, L_optimal_nFeatures]=min(Lin.MeanValidationErrorStorage);
    [dQ_lowestMeanValidationError, dQ_optimal_nFeatures]=min(DiagQuad.MeanValidationErrorStorage);
    
    DiagLin.OptimalMeanValidationErrorStorage=dL_lowestMeanValidationError;
    Lin.OptimalMeanValidationErrorStorage=L_lowestMeanValidationError;
    DiagQuad.OptimalMeanValidationErrorStorage=dQ_lowestMeanValidationError;
    
    DiagLin.OptimalHyperParamStorage=dL_optimal_nFeatures;
    Lin.OptimalHyperParamStorage=L_optimal_nFeatures;
    DiagQuad.OptimalHyperParamStorage=dQ_optimal_nFeatures;
    
    