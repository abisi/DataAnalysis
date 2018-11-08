%% Nested cross-validation; feature ranking using foward feature selection ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%reducing data dimensionnality by a factor of 10
trainData=trainData(1:10:end,:);
trainLabels=trainLabels(1:10:end,:);

%priors struct:  we need to specify priors when fitting models to data
%where class frequencies are unknown
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

%outer/inner fold counts
kOuter=3;
kInner=4;

%observation count, upper threashold for nbFeatures hyperParam
nObservations=length(trainLabels);

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
    
    %selection criteria & ffs parameters
    dL_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diaglinear', 'Prior', Priors), xt)));
    L_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear', 'Prior', Priors), xt)));
    dQ_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagquadtratic', 'Prior', Priors), xt)));
    Q_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'quadratic', 'Prior', Priors), xt)));
    opt = statset('Display','iter','MaxIter',100);
    
    %partition for inner cross validation
    cp=cvpartition(nOuterTrainingSet,'kfold',kInner);

    %run foward feature selection
    [dL_sel,dL_hst] = sequentialfs(dL_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [L_sel,L_hst] = sequentialfs(L_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [dQ_sel,dQ_hst] = sequentialfs(dQ_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [Q_sel,Q_hst] = sequentialfs(Q_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    
    %store optimal features for each model
    DiagLin.OptimalFeatures=find(dL_sel);
    Lin.OptimalFeatures=find(L_sel);
    DiagQuad.OptimalFeatures=find(dQ_sel);
    Quad.OptimalFeatures=find(Q_sel);
    
    %store optimal validation error for each model
    DiagLin.OptimalValidationError=dL_hst.Crit(end);
    Lin.OptimalValidationError=L_hst.Crit(end);
    DiagQuad.OptimalValidationError=dQ_hst.Crit(end);
    Quad.OptimalValidationError=Q_hst.Crit(end);
    
    %build optimal models
    optimalDiagLinearClassifier = fitcdiscr(outerTrainingSet(:,DiagLin.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    optimalLinearClassifier = fitcdiscr(outerTrainingSet(:,Lin.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'linear','Prior',Priors);
    optimalDiagQuadraticClassifier = fitcdiscr(outerTrainingSet(:,DiagQuad.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'diagquadratic','Prior',Priors);
    optimalQuadraticClassifier = fitcdiscr(outerTrainingSet(:,Quad.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'quadratic','Prior',Priors);
  
    %compute predicted classifications for test section data using optimal
    %models
    dL_prediction=predict(optimalDiagLinearClassifier, testSet(:,DiagLin.OptimalFeatures));
    L_prediction=predict(optimalLinearClassifier, testSet(:,Lin.OptimalFeatures));
    dQ_prediction=predict(optimalDiagQuadraticClassifier, testSet(:,DiagQuad.OptimalFeatures));
    Q_prediction=predict(optimalQuadraticClassifier, testSet(:,Quad.OptimalFeatures));
    
    %compute and store test errors for each model
    DiagLin.TestError = computeClassError(testLabels,dL_prediction);
    Lin.TestError = computeClassError(testLabels,L_prediction);
    DiagQuad.TestError = computeClassError(testLabels,dQ_prediction);
    Quad.TestError = computeClassError(testLabels,Q_prediction);
    
  
end