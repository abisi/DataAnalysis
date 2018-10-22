load('trainlabels.mat');
load('trainLabels.mat');

C = cvpartition(597, 'kFold', 10) %trainData(:,750), n=597
C.NumTestSets;

err=zeros(C.NumTestSets, 1);

for i=1:C.NumTestSets
    trIdx=C.TrainSize(i);
    teIdx=C.TestSize(i);
    ytest=classify(trainData(teIdx,:), trainData(trIdx,:), trainLabels(trIdx,:));
    err(i)=sum(~strcmp(ytest,trainLabels(teIdx)));
end

cvErr=sum(err)/(sum(C.TestSize));






%Matlab template
% load('fisheriris');
%       CVO = cvpartition(species,'KFold',10);
%       err = zeros(CVO.NumTestSets,1);
%       for i = 1:CVO.NumTestSets
%           trIdx = CVO.training(i);
%           teIdx = CVO.test(i);
%           ytest = classify(meas(teIdx,:),meas(trIdx,:),species(trIdx,:));
%           err(i) = sum(~strcmp(ytest,species(teIdx)));
%       end
%       cvErr = sum(err)/sum(CVO.TestSize);