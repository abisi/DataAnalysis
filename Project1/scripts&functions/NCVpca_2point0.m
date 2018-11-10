%% PCA nested cross-validation 
% Aim find best hyperparameters (number of PCs and classifier)
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Simple corss-validation loop with fix classifier for now! (diaglin)
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

kout=4;
kin=5;
nObservations=length(trainLabels);
Nmax=200; %number of PCs

%Error resulting from the inner loop ('simple cross-validation')
diaglin.trainErrorStorage=zeros(Nmax,kin); %inner error
diaglin.validationErrorStorage=zeros(Nmax,kin);

lin.trainErrorStorage=zeros(Nmax, kin);
lin.validationErrorStorage=zeros(Nmax, kin);

diagquad.trainErrorStorage=zeros(Nmax, kin);
diagquad.validationErrorStorage=zeros(Nmax, kin);

%MEAN train Error + mean validation error
diaglin.meanTrainError=zeros(Nmax, 1); %inner 
diaglin.meanValidationError=zeros(Nmax, 1); 

lin.meanTrainError=zeros(Nmax, 1); %inner 
lin.meanValidationError=zeros(Nmax, 1); %il faudrait mettre (Nmax, kout) comme c'est un vecteur. A vérifier a la compil!

diagquad.meanTrainError=zeros(Nmax, 1); %inner 
diagquad.meanValidationError=zeros(Nmax, 1);

%Erreur de training + validation MIN > forcément il y en a une pas outer fold
diaglin.minMeanTrainError=zeros(1, kout);
diaglin.minMeanValidationError=zeros(1, kout);

lin.minMeanTrainError=zeros(1, kout);
lin.minMeanValidationError=zeros(1, kout);

diagquad.minMeanTrainError=zeros(1, kout);
diagquad.minMeanValidationError=zeros(1, kout);


%Erreur finale calculée pour évaluer la PERF du modèle (!!!)
diaglin.testErrorStorage=zeros(1,kout);

lin.testErrorStorage=zeros(1, kout);

diagquad.testErrorStorage=zeros(1, kout);


%Nombre de components N optimal
diaglin.optimalComponentsNumber=zeros(1, kout);
lin.optimalComponentsNumber=zeros(1, kout);
diagquad.optimalComponentsNumber=zeros(1, kout);


%Normalize data before applying PCA
%trainData=zscore(trainData);

%Time for PCA
[coeff, score, variance]=pca(trainData); %As we're working with transformed features, we have to do it before

%Normalize the transformed data (after PCA)
score=zscore(score);

partitionOut=cvpartition(nObservations, 'kfold', kout); %same dimension of score and trainLabels (597 lines)

for i=1:kout
    %Indexes
    outerTrainIds=partitionOut.training(i);
    outerTestIds=partitionOut.test(i);
    
    %Outer train (2/3)
    outerTrainSet=score(outerTrainIds, :); %trainData
    outerTrainLabels=trainLabels(outerTrainIds, :);
    
    %Outer test (1/3)
    outerTestSet=score(outerTestIds, :); %lui on le garde pour la fin %trainData
    outerTestLabels=trainLabels(outerTestIds, :);
    
    %To partition the inner set we need to get the size of all the inner fold
    sizeOuterTrain=size(outerTrainSet,1);
    partitionIn = cvpartition(sizeOuterTrain, 'kfold', kin); %les data transformés ont toujours 597 observations

    for t=1:kin        
        innerTrainIds=partitionIn.training(t);
        innerValidationIds=partitionIn.test(t);
        
        %Inner train set + labels
        innerTrainSet=outerTrainSet(innerTrainIds, :);
        innerTrainLabels=outerTrainLabels(innerTrainIds, :);
        
        %Validation set + labels
        innerValidationSet=outerTrainSet(innerValidationIds, :); %inner test set = validation set
        innerValidationLabels=outerTrainLabels(innerValidationIds, :);
        
        %Time for PCA
        for N=1:Nmax                        
            selectedComponents=score(:, 1:N); %Components are classified by importance order.
            
            %diaglinear model + prediction
            diaglinClassifier = fitcdiscr(innerTrainSet(:, 1:N), innerTrainLabels, 'DiscrimType', 'diaglinear', 'Prior', Priors); 
            diaglinInnerTrainPrediction=predict(diaglinClassifier, innerTrainSet(:, 1:N));
            diaglinValidationPrediction=predict(diaglinClassifier, innerValidationSet(:, 1:N));
            
            %linear model + prediction
            linClassifier = fitcdiscr(innerTrainSet(:, 1:N), innerTrainLabels, 'DiscrimType', 'linear', 'Prior', Priors);
            linInnerTrainPrediction = predict(linClassifier, innerTrainSet(:, 1:N));
            linValidationPrediction = predict(linClassifier, innerValidationSet(:, 1:N));
            
            %diagquadratic model + prediction
            diagquadClassifier = fitcdiscr(innerTrainSet(:, 1:N), innerTrainLabels, 'DiscrimType', 'diagquad', 'Prior', Priors);
            diagquadInnerTrainPrediction = predict(diagquadClassifier, innerTrainSet(:, 1:N));
            diagquadValidationPrediction = predict(diagquadClassifier, innerValidationSet(:, 1:N));
            
            
            %Compute ERRORS associated to the inner loop ('simple cross-validation')
            %Diaglin
            diaglinInnerTrainError=computeClassError(diaglinInnerTrainPrediction, innerTrainLabels);
            diaglinValidationError=computeClassError(diaglinValidationPrediction, innerValidationLabels); %testError or validationError
            diaglin.trainErrorStorage(N, t)=diaglinInnerTrainError; %inner
            diaglin.validationErrorStorage(N, t)=diaglinValidationError;
           
            %lin
            linInnerTrainError=computeClassError(linInnerTrainPrediction, innerTrainLabels);
            linValidationError=computeClassError(linValidationPrediction, innerValidationLabels);
            lin.trainErrorStorage(N, t)=linInnerTrainError;
            lin.validationErrorStorage(N, t)=linValidationError;
           
            %diagquad
            diagquadInnerTrainError=computeClassError(diagquadInnerTrainPrediction, innerTrainLabels);
            diagquadValidationError=computeClassError(diagquadValidationPrediction, innerValidationLabels);
            diagquad.trainErrorStorage(N, t)=diagquadInnerTrainError;
            diagquad.validationErrorStorage(N, t)=diagquadValidationError;
        end
    end
    
    %Compute MEAN errors of TRAIN + VALIDATION and optimal Hyperparam
    %diaglin
    diaglin.meanTrainError=mean(diaglin.trainErrorStorage, 2); %problème: on écrase toujours la derière boucle comme .meaTrainError est de taille (N, 1). il faudrait (N, 3) non?
    diaglin.meanValidationError=mean(diaglin.validationErrorStorage,  2);
    
    [dl_optimalMeanValidationError, dl_optimalIdx]=min(diaglin.meanValidationError);
    
    diaglin.minMeanTrainError(1, i)=min(diaglin.meanTrainError);
    diaglin.minMeanValidationError(1, i)=dl_optimalMeanValidationError;
    
    diaglin.optimalComponentsNumber(1, i)=dl_optimalIdx;
    
    %lin
    lin.meanTrainError=mean(lin.trainErrorStorage, 2);
    lin.meanValidationError=mean(lin.validationErrorStorage, 2);
    
    [l_optimalMeanValidationError, l_optimalIdx]=min(lin.meanValidationError);
    
    lin.minMeanTrainError(1, i)=min(lin.meanTrainError);
    lin.minMeanValidationError(1, i)=l_optimalMeanValidationError;
    
    lin.optimalComponentsNumber(1, i)=l_optimalIdx;
    
    %diagquad
    diagquad.meanTrainError=mean(diagquad.trainErrorStorage, 2);
    diagquad.meanValidationError=mean(diagquad.validationErrorStorage, 2);
    
    [dq_optimalMeanValidationError, dq_optimalIdx]=min(diagquad.meanValidationError);
    
    diagquad.minMeanTrainError(1, i)=min(diagquad.meanTrainError);
    diagquad.minMeanValidationError(1, i)=dq_optimalMeanValidationError;
    
    diagquad.optimalComponentsNumber(1, i)=dq_optimalIdx;
    
    
    %EVALUATION OF OPTIMAL MODEL    
    %dl
    dl_optimalModel=fitcdiscr(outerTrainSet(:, 1:dl_optimalIdx), outerTrainLabels, 'DiscrimType', 'diaglinear','Prior',Priors);
    dl_optimalModelPrediction=predict(dl_optimalModel, outerTestSet(:, 1:dl_optimalIdx));
    
    %Compute errors associated to the outer loop = TEST ERRORS
    diaglinTestError=computeClassError(dl_optimalModelPrediction, outerTestLabels);
    diaglin.testErrorStorage(1, i)=diaglinTestError; 
    
    %lin
    l_optimalModel=fitcdiscr(outerTrainSet(:, 1:l_optimalIdx), outerTrainLabels, 'DiscrimType', 'linear','Prior',Priors);
    l_optimalModelPrediction=predict(l_optimalModel, outerTestSet(:, 1:l_optimalIdx));    
    
    linTestError=computeClassError(l_optimalModelPrediction, outerTestLabels);
    lin.testErrorStorage(1, i)=linTestError;
        
    %diagquad
    dq_optimalModel=fitcdiscr(outerTrainSet(:, 1:dq_optimalIdx), outerTrainLabels, 'DiscrimType', 'diagquad','Prior',Priors);
    dq_optimalModelPrediction=predict(dq_optimalModel, outerTestSet(:, 1:dq_optimalIdx));
    
    diagquadTestError=computeClassError(dq_optimalModelPrediction, outerTestLabels);
    diagquad.testErrorStorage(1, i)=diagquadTestError;

end


