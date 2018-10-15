%% LDA/QDA classifiers
featuresWhole=trainData;
featuresSubset=trainData(:,1:10:end);
labels=trainLabels;

% Using subset of features:
%Non-uniform classes
classifierDiagLin=fitcdiscr(featuresSubset,labels,'DiscrimType','linear');
%Uniform classes
classifierLinUnif=fitcdiscr(featuresSubset,labels,'DiscrimType','linear','prior','uniform');
classifierDiagLin=fitcdiscr(featuresSubset,labels,'DiscrimType','diaglinear','prior','uniform');
%classifierQuad=fitcdiscr(featuresSubset,labels,'DiscrimType','quadratic'); %covariance matrix SINGULAR 
classifierDiagQuad=fitcdiscr(featuresSubset,labels,'DiscrimType','diagquadratic','prior','uniform')

%Predictions 
yhatLin=predict(classifierDiagLin,featuresSubset);
yhatLinUnif=predict(classifierLinUnif,featuresSubset); %Uniform
yhatDiagLin=predict(classifierDiagLin,featuresSubset);
%yhatQuad=predict(classifierQuad,featuresSubset);
yhatDiagQuad=predict(classifierDiagQuad,featuresSubset);

%Classification error calculations for each classifier:
correctCounterLin=0;
correctCounterLinUnif=0; %Uniform
correctCounterDiagLin=0;
%correctCounterQuad=0;
correctCounterDiagQuad=0;

for i=1:597
    if(yhatLin(i,1)==trainLabels(i,1))
        correctCounterLin=correctCounterLin+1;
    end
    if(yhatLinUnif(i,1)==trainLabels(i,1))
        correctCounterLinUnif=correctCounterLinUnif+1;
    end
     if(yhatDiagLin(i,1)==trainLabels(i,1))
        correctCounterDiagLin=correctCounterDiagLin+1;
     end
     %if(yhatQuad(i,1)==trainLabels(i,1))
      %  correctCounterQuad=correctCounterQuad+1;
     %end
     if(yhatDiagQuad(i,1)==trainLabels(i,1))
        correctCounterDiagQuad=correctCounterDiagQuad+1;
    end
end

classificationAccuracyLin=correctCounterLin/597;
classificationErrorLin=1-classificationAccuracyLin;

classificationAccuracyLinUnif=correctCounterLinUnif/597;
classificationErrorLinUnif=1-classificationAccuracyLinUnif;

classificationAccuracyDiagLin=correctCounterDiagLin/597;
classificationErrorDiagLin=1-classificationAccuracyDiagLin;

%classificationAccuracyQuad=correctCounterQuad/597;
%classificationErrorQuad=1-classificationAccuracyQuad;

classificationAccuracyDiagQuad=correctCounterDiagQuad/597;
classificationErrorDiagQuad=1-classificationAccuracyDiagQuad;

%Error typing
corrError=0;
errError=0;
for i=1:597
    if(yhatLin(i)~=trainLabels(i,1) && trainLabels(i,1)==0)
        corrError=corrError+1;
    else if (yhatLin(i)~=trainLabels(i,1) && trainLabels(i,1)==1)
            errError=errError+1;
        end
    end
end

% Class error calculations (linear classifier only):
numberErr=nnz(trainLabels);
numberCorr=597-numberErr;
classError=0.5*(errError/numberErr)+0.5*(corrError/numberCorr);


% Error typing
corrError=0;
errError=0;
for i=1:597
    if(yhatLinUnif(i)~=trainLabels(i,1) && trainLabels(i,1)==0)
        corrError=corrError+1;
    else if (yhatLinUnif(i)~=trainLabels(i,1) && trainLabels(i,1)==1)
            errError=errError+1;
        end
    end
end
% Class error calculations (linear classifier only) w/ uniform argument:
numberErr=nnz(trainLabels);
numberCorr=597-numberErr;
classErrorUnif=0.5*(errError/numberErr)+0.5*(corrError/numberCorr);


%% Training and testing error
% From previous section, we're working with : class error
% Divide dataset
set1 = trainData(1:2:end,1:10:end);
set2 = trainData(2:2:end,1:10:end);
label1 = trainLabels(1:2:end);
label2 = trainLabels(2:2:end);

ratio = 0.33;
% Comparison train and test errors
classifierDiagLin = fitcdiscr(set1,label1,'DiscrimType','diaglinear');
predictionDiagLin = predict(classifierDiagLin, set1);

trainingError = computeClassError(label1, predictionDiagLin, ratio);
testingError = computeClassError(label2, predictionDiagLin, ratio);
errorDiagLin = [trainingError, testingError]

%Same with linear, diagquadratic, quadratic
%Linear
classifierLin = fitcdiscr(set1,label1,'DiscrimType','linear');
predictionLin = predict(classifierLin, set1);

trainingError = computeClassError(label1, predictionLin, ratio);
testingError = computeClassError(label2, predictionLin, ratio);
errorLin = [trainingError, testingError]

%Diagquadratic
classifierDiagQuad = fitcdiscr(set1,label1,'DiscrimType','diagquadratic');
predictionDiagQuad = predict(classifierDiagQuad, set1);

trainingError = computeClassError(label1, predictionDiagQuad, ratio);
testingError = computeClassError(label2, predictionDiagQuad, ratio);
errorDiagQuad = [trainingError, testingError]

%Quadratic - isn't supposed to work
%classifierQuad = fitcdiscr(set1,trainLabels(1:2:end),'DiscrimType','quadratic');
%predictionQuad = predict(classifierQuad, set1);

%trainingError = computeClassError(label1, predictionQuad, ratio);
%testingError = computeClassError(label2, predictionQuad, ratio);
%errorQuad = [trainingError, testingError]

%- Would we still choose the same classifier ? -> linear has errors ~= 0.03
%- Improvement on training error does not improve testing error as
%- Can't use quadratic : covariance matrix is SINGULAR i.e. not invertible

%Complexity - number of parameters
%Diaglinear:
%Linear:
%Diagquadratic:
%Number of training samples:
%- Are these classifiers robust ?


%% Again but revert set1 & set2 : training on set 2
%Notive the performance variability 



%% Cross validation for performance estimation

N = length(trainData);
k = 10; %or we could set it up to 6 as we have ~600 samples, so size(subset)=100...?

%Partition comparison
cp_N = cvpartition(N, 'kfold', k);
cp_labels = cvpartition(trainLabels, 'kfold', k);

