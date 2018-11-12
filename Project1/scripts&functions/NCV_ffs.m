%% Nested cross-validation; feature ranking using foward feature selection ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Reducing data dimensionality by a factor of 5 - sequentialfs takes time
trainData=trainData(:,1:5:end);
trainLabels=trainLabels(:,1:5:end);

trainData=zscore(trainData);

%Outer/inner fold counts
kOuter=3;
kInner=4;

%Optimal features per classyfier type
DiagLin.OptimalFeatures={};
Lin.OptimalFeatures={};
DiagQuad.OptimalFeatures={};
Quad.OptimalFeatures={};

%Storage for optimal validation error
DiagLin.OptimalValidationError=zeros(1,kOuter);
Lin.OptimalValidationError=zeros(1,kOuter);
DiagQuad.OptimalValidationError=zeros(1,kOuter);
Quad.OptimalValidationError=zeros(1,kOuter);

%Storage for test error
DiagLin.TestError=zeros(1,kOuter);
Lin.TestError=zeros(1,kOuter);
DiagQuad.TestError=zeros(1,kOuter);
Quad.TestError=zeros(1,kOuter);

%Observation count
nObservations=length(trainLabels);

%Break total data into 4 outer folds
partition_outer = cvpartition(nObservations, 'kfold', kOuter);

%For each outer fold cycle
for i=1:kOuter
    disp('Outer fold ' + num2str(i))
    %get training section indexes
    outerTrainingMarker=partition_outer.training(i);
    %get test section indexes
    testMarker=partition_outer.test(i);
    
    %load training section data
    outerTrainingSet=trainData(outerTrainingMarker, :);
    %load training section labels
    outerTrainingLabels=trainLabels(outerTrainingMarker, :);
    
    %load test section data
    testSet=trainData(testMarker, :); 
    %load test section labels
    testLabels=trainLabels(testMarker, :);
    
    %size of training section (in terms of nObservations)
    nOuterTrainingSet=size(outerTrainingSet,1);
    
    %selection criteria & ffs parameters
    dL_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagLinear','Prior','uniform'), xt)));
    L_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear', 'Prior','uniform'), xt)));
    dQ_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagquadratic', 'Prior','uniform'), xt)));
    Q_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'quadratic', 'Prior','uniform'), xt)));
    opt = statset('Display','iter','MaxIter',100);
    
    %Partition for inner cross validation
    cp=cvpartition(nOuterTrainingSet,'kfold',kInner);

    %Run foward feature selection
    disp('Diaglinear')
    [dL_sel,dL_hst] = sequentialfs(dL_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',kInner,'options',opt);
    disp('Linear')
    [L_sel,L_hst] = sequentialfs(L_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',kInner,'options',opt);
    disp('Diagquadratic')
    [dQ_sel,dQ_hst] = sequentialfs(dQ_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',kInner,'options',opt);
    disp('Quadratic')
    [Q_sel,Q_hst] = sequentialfs(Q_selectionCriteria,outerTrainingSet,outerTrainingLabels,'cv',kInner,'options',opt);
    
    %Compute and save optimal features for each model
    DiagLin.OptimalFeatures{i}=find(dL_sel);
    Lin.OptimalFeatures{i}=find(L_sel);
    DiagQuad.OptimalFeatures{i}=find(dQ_sel);
    Quad.OptimalFeatures{i}=find(Q_sel);
    
    %Compute and save optimal validation error for each model
    DiagLin.OptimalValidationError(1,i)=dL_hst.Crit(end);
    Lin.OptimalValidationError(1,i)=L_hst.Crit(end);
    DiagQuad.OptimalValidationError(1,i)=dQ_hst.Crit(end);
    Quad.OptimalValidationError(1,i)=Q_hst.Crit(end);
    
    %Build optimal models
    optimalDiagLinearClassifier = fitcdiscr(outerTrainingSet(:,DiagLin.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'diaglinear');
    optimalLinearClassifier = fitcdiscr(outerTrainingSet(:,Lin.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'linear');
    optimalDiagQuadraticClassifier = fitcdiscr(outerTrainingSet(:,DiagQuad.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'diagQuadratic');
    optimalQuadraticClassifier = fitcdiscr(outerTrainingSet(:,Quad.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'quadratic');
  
    %Compute predicted classifications for test section data using optimal
    %models
    dL_prediction=predict(optimalDiagLinearClassifier, testSet(:,DiagLin.OptimalFeatures{i}));
    L_prediction=predict(optimalLinearClassifier, testSet(:,Lin.OptimalFeatures{i}));
    dQ_prediction=predict(optimalDiagQuadraticClassifier, testSet(:,DiagQuad.OptimalFeatures{i}));
    Q_prediction=predict(optimalQuadraticClassifier, testSet(:,Quad.OptimalFeatures{i}));
    
    %Compute and save test errors for each model
    DiagLin.TestError(1,i) = computeClassError(testLabels,dL_prediction);
    Lin.TestError(1,i) = computeClassError(testLabels,L_prediction);
    DiagQuad.TestError(1,i) = computeClassError(testLabels,dQ_prediction);
    Quad.TestError(1,i) = computeClassError(testLabels,Q_prediction);
   
  
end




