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
ratio=0.5;


fun = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', classifiertype), xt), ratio));

opt = statset('Display','iter','MaxIter',100);
cp=cvpartition(trainLabels(1:10:end, :),'kfold',k);
[sel,hst] = sequentialfs(fun,trainData(1:10:end,:),trainLabels(1:10:end, :),'cv',cp,'options',opt);



