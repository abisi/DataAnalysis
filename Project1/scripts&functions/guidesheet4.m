
%% Guidesheet 4: Principle Component Analysis
[coeff,score,variance]=pca(trainData);

priorCov=cov(trainData);
postCov=cov(score);

diagPriorCov=diag(priorCov);
diagPostCov=diag(postCov);


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



