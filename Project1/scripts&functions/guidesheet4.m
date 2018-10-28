clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
%% Guidesheet 4: Principle Component Analysis
[coeff,score,variance]=pca(trainData);

priorCov=cov(trainData);
postCov=cov(score);

diagPriorCov=diag(priorCov);
diagPostCov=diag(postCov);