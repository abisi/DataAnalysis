%% PCA and cross-validation
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');

%Simple cross-validation loop with fix classifier for now! (diaglin)
Priors.ClassNames=[0 1];
Priors.ClassProbs=[0.7 0.3];
k=10;
nObservations=length(trainLabels);
Nmax=100; % max number of PCs 

trainErrorStorage=zeros(Nmax,k);
testErrorStorage=zeros(Nmax,k);
optimalHyperParamStorage=0; %number of Principal components to obtain 90% total variance.

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
        
        %Nmax=size(score, 2);
        
        for N=1:Nmax                                   
            classifier = fitcdiscr(score(:, 1:N), trainingLabels, 'DiscrimType', 'diaglinear', 'Prior', Priors); 
            
            trainPrediction=predict(classifier, score(:, 1:N));
            
            scoreTest = (testSetNorm-mean(testSetNorm, 1)) * coeff;
            testPrediction=predict(classifier, scoreTest(:, 1:N));
            
            
            trainError=computeClassError(trainPrediction, trainingLabels);
            testError=computeClassError(testPrediction, testLabels);
            
            trainErrorStorage(N,t)=trainError;
            testErrorStorage(N,t)=testError;
        end
end

%Compute average errors > mean makes the mean on columns
meanTrainError=(mean(trainErrorStorage'))';
meanTestError=(mean(testErrorStorage'))';

min(meanTestError)

figure;
plot(1:Nmax, meanTrainError, 'g', 1:Nmax, meanTestError, 'r');
xlabel('N'); %number of principal components
ylabel('Error');
legend('Train error', 'Test error');