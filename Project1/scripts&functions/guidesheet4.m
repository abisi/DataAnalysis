clear all
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
%% Guidesheet 4: Principle Component Analysis
[coeff,score,variance]=pca(trainData);

priorCov=cov(trainData);
postCov=cov(score);

diagPriorCov=diag(priorCov);
diagPostCov=diag(postCov);


%% Forward Feature Selection
classifiertype='diaglinear';
k=10;

fun = @(xT,yT,xt,yt) length(yt)*(your_error(yt,predict(fitcdiscr(xT,yT,'discrimtype', classifiertype), xt)));

opt = statset('Display','iter','MaxIter',100);
cp=cvpartition(trainLabels,'kfold',k);
[sel,hst] = sequentialfs(fun,trainData,trainLabels,'cv',cp,'options',opt);



