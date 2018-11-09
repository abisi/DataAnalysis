%% Guidesheet 4: Principle Component Analysis
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

% Principal Component Analysis
[coeff,score,variance]=pca(trainData);

priorCov=cov(trainData);
postCov=cov(score);
%figure;
%imshow(priorCov, [0, 0.01]);
figure;
imagesc(priorCov, [0, 0.01]);
colorbar;
title('Covariance matrix of the train data');

figure;
imagesc(postCov, [0, 3e-18]);
title('Covariance matrix of the transformed data');

%We can observe the 16 EEG channels (clearly separated on the cov matrix).
%The pattern give us information about the correlation between samples.

diagPriorCov=diag(priorCov);
diagPostCov=diag(postCov);

meanPriorVar=mean(diagPriorCov);
meanPostVar=mean(diagPostCov);

%Maximum covariance values original data vs transformed data
priorOffDiag=priorCov-diag(diagPriorCov);
maxPriorCov=max(max(priorOffDiag));

postOffDiag=postCov-diag(diagPostCov);
maxPostCov=max(max(postOffDiag));
% we can notice the (huge) decrease in covariance between the 2 (0.0360
% against 1.43e-15)

%PCA maximises the variance in order to get rid off low-variance dimensions.
%Therefore, we observe that the diagonal of the data (before and after
%tranformation), which corresponds to variance, is larger for the
%transformed data (referred as post).
%In terms of informative power, it means that the information content carried by
%the projected data is higher (i.e. lower entropy).

%Diagonal spread along eigenvectors is expressed by the covariance. The
%covariance is minimized in the transformed features. The correlation is
%thus minimised as well, meaning that the features are not correlated and
%carry maximum information. Each PC represent a decrease in the system's
%entropy.

% PCs as Hyperparameters: 
cumVar=cumsum(variance)/sum(variance);
numPC=1:length(variance);
figure;
plot(numPC, cumVar, 'r'); hold on;
xlabel('Number of PCs');
ylabel('Percentage of the total variance');
title('Total information carried by Principal Components');

idx90=find(cumVar>0.9);
pc90=numPC(idx90(1));
threshold90=line([pc90 pc90], [0 1]);
set(threshold90,'LineWidth',1,'color','blue');

%Let's find minimum distance 
%Point of 'diminishing returns': where little variance is gained
distance = sqrt((numPC - zeros(1,length(numPC))).^2 + (cumVar' - ones(1, length(cumVar))).^2);
[minDistance, idxDistance] = min(distance);

%% PCA and cross-validation
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Simple cross-validation loop with fix classifier for now! (diaglin)
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
k=5;
nObservations=length(trainLabels);
Nmax=250; %number of PCs

trainErrorStorage=zeros(Nmax,k);
testErrorStorage=zeros(Nmax,k);
optimalHyperParamStorage=0; %number of Principal components to obtain 90% total variance.

[coeff, score, variance]=pca(trainData); %As we're working with transformed features, we have to do it before

    for t=1:k
        partition = cvpartition(nObservations, 'kfold', k); %les data transformés ont toujours 597 observations
        
        trainMarker=partition.training(t);
        testMarker=partition.test(t);
        
        trainSet=score(trainMarker,:); %new features
        trainingLabels=trainLabels(trainMarker, :); %vrais labels associés ne changent pas.     
        
        testSet=score(testMarker, :);
        testLabels=trainLabels(testMarker,:);
        
        for N=1:Nmax                        
            selectedComponents=score(:, 1:N); %Components are classified by importance order.
                       
            classifier = fitcdiscr(trainSet(:, 1:N), trainingLabels, 'DiscrimType', 'diaglinear', 'Prior', Priors); 
            
            trainPrediction=predict(classifier, trainSet(:, 1:N));
            testPrediction=predict(classifier, testSet(:, 1:N));
                   
            trainError=computeClassError(trainPrediction, trainingLabels);
            testError=computeClassError(testPrediction, testLabels);
            
            trainErrorStorage(N,t)=trainError;
            testErrorStorage(N,t)=testError;
        end
    end

%Compute average errors > mean makes the mean on columns
meanTrainError=(mean(trainErrorStorage'))';
meanTestError=(mean(testErrorStorage'))';

figure;
plot(1:Nmax, meanTrainError, 'g', 1:Nmax, meanTestError, 'r');
xlabel('N'); %number of principal components
ylabel('Error');
legend('Train error', 'Test error');


%% PCA model - all the data
%Let's say
N = 108
%Missing data
augmentedData = vertcat(trainData, trainData, trainData, trainData);
%PCA
[coeff, score, variance]=pca(augmentedData); %As we're working with transformed features, we have to do it before
for i=1:length(coeff)
    if i > N 
        coeff(i) = 0;
    end
end

dataInPCASpace = coeff*augmentedData;
classifier = fitcdiscr(dataInPCASpace, labels, 'DiscrimType','diaglin');

%% Forward Feature Selection
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

classifiertype='diaglinear';
k=10;
ratio=0.5;


selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', classifiertype, 'Prior', Priors), xt)));
opt = statset('Display','iter','MaxIter',100);
cp=cvpartition(trainLabels(1:10:end, :),'kfold',k);

[sel,hst] = sequentialfs(selectionCriteria,trainData(1:10:end,:),trainLabels(1:10:end, :),'cv',cp,'options',opt);

%% Nested cross-valiation using FFF instead of rankfeat

clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

kOut=3;
kIn=10;
partitionOut=cvpartition(length(trainLabels), 'kfold', kOut);
    






