%% Guidesheet 4: Principle Component Analysis
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

% Principal Component Analysis
[coeff,score,variance]=pca(trainData);

priorCov=cov(trainData);
postCov=cov(score);
%imshow(postCov);

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

%PCA maximise the variance in order to get rid off low-variance dimensions.
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

% PCs as Hyperparameters
cumVar=cumsum(variance)/sum(variance);
numPC=1:length(variance);
plot(numPC, cumVar, 'r'); hold on;
xlabel('Number of PCs');
ylabel('Percentage of the total variance');
title('Total information carried by Principal Components');

idx90=find(cumVar>0.9);
pc90=numPC(idx90(1));
threshold90=line([pc90 pc90], [0 1]);
set(threshold90,'LineWidth',1,'color','blue');


%% PCA and cross-validation



%% Forward Feature Selection
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

classifiertype='diaglinear';
k=10;
ratio=0.5;


selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', classifiertype, 'Prior', Priors), xt), ratio));
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
    






