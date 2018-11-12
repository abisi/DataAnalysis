clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

trainData=trainData(:,1:10:end);
trainLabels=trainLabels(:,1:10:end);
nObservations=length(trainLabels);

Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];

DiagLin.OptimalFeatures=[];
Lin.OptimalFeatures=[];
DiagQuad.OptimalFeatures=[];
Quad.OptimalFeatures=[];

DiagLin.OptimalValidationError=0;
Lin.OptimalValidationError=0;
DiagQuad.OptimalValidationError=0;
Quad.OptimalValidationError=0;

k=5;

cp=cvpartition(nObservations,'kfold',k);
opt = statset('Display','iter','MaxIter',100);

dL_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diaglinear','Prior','uniform'), xt)));
L_selectionCriteria = @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'linear','Prior','uniform'), xt)));
dQ_selectionCriteria= @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'diagQuadratic', 'Prior','uniform'), xt)));
Q_selectionCriteria= @(xT,yT,xt,yt) length(yt)*(computeClassError(yt,predict(fitcdiscr(xT,yT,'discrimtype', 'quadratic', 'Prior','uniform'), xt)));

[dL_sel,dL_hst] = sequentialfs(dL_selectionCriteria,trainData,trainLabels,'cv',k,'options',opt);
[L_sel,L_hst] = sequentialfs(L_selectionCriteria,trainData,trainLabels,'cv',k,'options',opt);
[dQ_sel,dQ_hst] = sequentialfs(dQ_selectionCriteria,trainData,trainLabels,'cv',k,'options',opt);
[Q_sel,Q_hst] = sequentialfs(Q_selectionCriteria,trainData,trainLabels,'cv',k,'options',opt);

DiagLin.OptimalFeatures=find(dL_sel);
Lin.OptimalFeatures=find(L_sel);
DiagQuad.OptimalFeatures=find(dQ_sel);
Quad.OptimalFeatures=find(Q_sel);

DiagLin.OptimalValidationError=dL_hst.Crit(end);
Lin.OptimalValidationError=L_hst.Crit(end);
DiagQuad.OptimalValidationError=dQ_hst.Crit(end);
Quad.OptimalValidationError=Q_hst.Crit(end);