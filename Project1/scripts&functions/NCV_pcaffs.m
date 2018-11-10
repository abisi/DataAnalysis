%% Nested cross-validation; feature ranking using foward feature selection ; hyperParams: nbFeatures, modelType

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Reducing data dimensionality by a factor of 5 - sequentialfs takes time
trainData=trainData(:,1:5:end);
trainLabels=trainLabels(:,1:5:end);

%Priors struct:  we need to specify priors when fitting models to data
%where class frequencies are unknown
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

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
    Disp('Outer fold' i)
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
    dL_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagLinear', 'Prior', Priors), xt)));
    L_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear', 'Prior', Priors), xt)));
    dQ_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagquadratic', 'Prior', Priors), xt)));
    Q_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'quadratic', 'Prior', Priors), xt)));
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
    optimalDiagLinearClassifier = fitcdiscr(outerTrainingSet(:,DiagLin.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    optimalLinearClassifier = fitcdiscr(outerTrainingSet(:,Lin.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'linear','Prior',Priors);
    optimalDiagQuadraticClassifier = fitcdiscr(outerTrainingSet(:,DiagQuad.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'diagQuadratic','Prior',Priors);
    optimalQuadraticClassifier = fitcdiscr(outerTrainingSet(:,Quad.OptimalFeatures{i}), outerTrainingLabels, 'DiscrimType', 'quadratic','Prior',Priors);
  
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

% Choice of hyperparameters (Nsel, model type)
meanModelErrors = [mean(DiagLin.TestError), mean(Lin.TestError), mean(DiagQuad.TestError), mean(Quad.TestError)];
disp(meanModelErrors)
%Model type: Quadratic 
%Nsel:{[10    11   110   128   131   133   174   187   202   204]} 

% Statistical significance - assuming error is normally distributed with
% unknown variance
[h, pvalue] = ttest(Quad.TestError, 0.5); 

%Since pvalue < 0.05, we can rejet the null hypothesis (0.5).




