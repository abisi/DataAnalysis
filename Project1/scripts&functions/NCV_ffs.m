%% Nested cross-validation; feature ranking using foward feature selection ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%reducing data dimensionnality by a factor of 10
trainData=trainData(:,1:10:end);
trainLabels=trainLabels(:,1:10:end);

%priors struct:  we need to specify priors when fitting models to data
%where class frequencies are unknown
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

%outer/inner fold counts
kOuter=3;
kInner=4;

%Storage for features of optimal model determined using foward feature
%selection on training section of outer partition. Vector of length kOuter
%(one entry per outer fold cycle).
DiagLin.OptimalFeatures=zeros(1,kOuter);
Lin.OptimalFeatures=zeros(1,kOuter);
DiagQuad.OptimalFeatures=zeros(1,kOuter);
Quad.OptimalFeatures=zeros(1,kOuter);

%Storage for validation error of optimal model determined using foward
%feature selection on training section of outer partition. Vector of length
%kOuter (one optimal validation error per outer fold cycle).
DiagLin.OptimalValidationError=zeros(1,kOuter);
Lin.OptimalValidationError=zeros(1,kOuter);
DiagQuad.OptimalValidationError=zeros(1,kOuter);
Quad.OptimalValidationError=zeros(1,kOuter);

%Storage for test error of optimal model determined using foward feature
%selection on training section of outer partition. Model is trained on
%outer training section data and tested on outer test section data.Vector
%of length kOuter (one entry per outer fold cycle).
DiagLin.TestError=zeros(1,kOuter);
Lin.TestError=zeros(1,kOuter);
DiagQuad.TestError=zeros(1,kOuter);
Quad.TestError=zeros(1,kOuter);

%observation count
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
    dL_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagLinear', 'Prior', Priors), xt)));
    L_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear', 'Prior', Priors), xt)));
    dQ_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagQuadratic', 'Prior', Priors), xt)));
    Q_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'quadratic', 'Prior', Priors), xt)));
    opt = statset('Display','iter','MaxIter',100);
    
    %partition for inner cross validation
    cp=cvpartition(nOuterTrainingSet,'kfold',kInner);

    %run foward feature selection
    [dL_sel,dL_hst] = sequentialfs(dL_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [L_sel,L_hst] = sequentialfs(L_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [dQ_sel,dQ_hst] = sequentialfs(dQ_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    [Q_sel,Q_hst] = sequentialfs(Q_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',cp,'options',opt);
    
    %compute and save optimal features for each model
    DiagLin.OptimalFeatures(1,i)=find(dL_sel);
    Lin.OptimalFeatures(1,i)=find(L_sel);
    DiagQuad.OptimalFeatures(1,i)=find(dQ_sel);
    Quad.OptimalFeatures(1,i)=find(Q_sel);
    
    %compute and save optimal validation error for each model
    DiagLin.OptimalValidationError(1,i)=dL_hst.Crit(end);
    Lin.OptimalValidationError(1,i)=L_hst.Crit(end);
    DiagQuad.OptimalValidationError(1,i)=dQ_hst.Crit(end);
    Quad.OptimalValidationError(1,i)=Q_hst.Crit(end);
    
    %build optimal models
    optimalDiagLinearClassifier = fitcdiscr(outerTrainingSet(:,DiagLin.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    optimalLinearClassifier = fitcdiscr(outerTrainingSet(:,Lin.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'linear','Prior',Priors);
    optimalDiagQuadraticClassifier = fitcdiscr(outerTrainingSet(:,DiagQuad.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'diagQuadratic','Prior',Priors);
    optimalQuadraticClassifier = fitcdiscr(outerTrainingSet(:,Quad.OptimalFeatures), outerTrainingLabels, 'DiscrimType', 'quadratic','Prior',Priors);
  
    %compute predicted classifications for test section data using optimal
    %models
    dL_prediction=predict(optimalDiagLinearClassifier, testSet(:,DiagLin.OptimalFeatures));
    L_prediction=predict(optimalLinearClassifier, testSet(:,Lin.OptimalFeatures));
    dQ_prediction=predict(optimalDiagQuadraticClassifier, testSet(:,DiagQuad.OptimalFeatures));
    Q_prediction=predict(optimalQuadraticClassifier, testSet(:,Quad.OptimalFeatures));
    
    %compute and save test errors for each model
    DiagLin.TestError(1,i) = computeClassError(testLabels,dL_prediction);
    Lin.TestError(1,i) = computeClassError(testLabels,L_prediction);
    DiagQuad.TestError(1,i) = computeClassError(testLabels,dQ_prediction);
    Quad.TestError(1,i) = computeClassError(testLabels,Q_prediction);
    
  
end