%% PCA and cross-validation
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Simple cross-validation loop with fix classifier for now! (diaglin)
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
k=5;
nObservations=length(trainLabels);
Nmax=476; % max number of PCs 

types=["linear","diaglinear","diagquadratic"];

trainErrorStorage=zeros(Nmax,k);
testErrorStorage=zeros(Nmax,k);

optimalHyperParamStorage=zeros(1, k); %number of Principal components to obtain 90% total variance.
error=zeros(1, k);
varianceStorage={};

varExplained=0.8;

partition = cvpartition(nObservations, 'kfold', k);


for t=1:k
        trainMarker=partition.training(t);
        testMarker=partition.test(t);
        
        trainSet=trainData(trainMarker,:); %new features
        trainingLabels=trainLabels(trainMarker); %vrais labels associés ne changent pas.     
        testSet=trainData(testMarker, :); %validation set
        testLabels=trainLabels(testMarker); %validation labels
        
        [trainSetNorm, muTrainSet, sigmaTrainSet]=zscore(trainSet);
        testSetNorm=(testSet-muTrainSet)./sigmaTrainSet;
        
        [coeff, score, variance]=pca(trainSetNorm);
        
        varianceStorage=cumsum(variance)/sum(variance)
        
        pcaDataTrain=trainSetNorm*coeff;
        pcaDataTest=testSetNorm*coeff;
        
        
        for type=1:length(types)
        
            for N=1:Nmax                                   
                classifier = fitcdiscr(pcaDataTrain(:, 1:N), trainingLabels, 'DiscrimType', types(type), 'prior', 'uniform'); 

                trainPrediction=predict(classifier, pcaDataTrain(:, 1:N));

                testPrediction=predict(classifier, pcaDataTest(:, 1:N));

                trainError=computeClassError(trainPrediction, trainingLabels);
                testError=computeClassError(testPrediction, testLabels);

                trainErrorStorage(N,t)=trainError;
                testErrorStorage(N,t)=testError;
            end 
        
        if types(type)=="linear"    
            l_optimalHyperparameterStorage(1, t)=min(find(varianceStorage>0.85));
            l_error(1, t)=testErrorStorage(min(find(varianceStorage>0.85)));
        end
        if types(type)=="diaglinear"  
            dl_optimalHyperparameterStorage(1, t)=min(find(varianceStorage>0.85));
            dl_error(1, t)=testErrorStorage(min(find(varianceStorage>0.85)));
        end
        if types(type)=="diagquadratic"  
            dq_optimalHyperparameterStorage(1, t)=min(find(varianceStorage>0.85));
            dq_error(1, t)=testErrorStorage(min(find(varianceStorage>0.85)));
        end
        
        end
  
end


%%

[minMeanTestError, minIdx]=min(mean(testErrorStorage, 2));
%varianceExplained=sum(variance(1:minIdx))/sum(variance);
varianceExplained=varianceStorage(minIdx,:);        
Noptimal=minIdx;

%%
meanTrainError=mean(trainErrorStorage,2);
meanTestError=mean(testErrorStorage, 2);
figure;
plot(1:Nmax, meanTrainError, 'g', 1:Nmax, meanTestError, 'r');
xlabel('N'); %number of principal components
ylabel('Error');
legend('Train error', 'Test error');

%%
types=["linear","diaglinear","diagquadratic", "quadratic"];
length(types)
