%% Guidesheet3: Model selection and nested cross-validation

%%Cross-validation for hyperparameter optimization

%Fisher-based feature selection
k = 10;
maxFt = 20;
cp_labels = cvpartition(trainLabels, 'kfold', k);
trainErrors = zeros(k, maxFt);
testErrors = zeros(k, maxFt);

for ft = 1:maxFt
    for i=1:cp_labels.NumTestSets
        trainId = cp_labels.training(i);
        testId = cp_labels.test(i);
        %Pick ft-th best features
    	[orderedInd, orderedPower] = rankfeat(trainData(trainId,:), trainLabels(trainId,:), 'fisher');
        bestInd = orderedInd(1:ft);
        %Build model and predict
        classifierDiagLin = fitcdiscr(trainData(trainId, bestInd), trainLabels(trainId,:), 'DiscrimType', 'diaglinear');
        yhatTrain = predict(classifierDiagLin, trainData(trainId, bestInd));
        yhatTest = predict(classifierDiagLin, trainData(testId, bestInd));
        %Compute both errors
        errTrain = computeClassificationError(trainLabels(trainId,:), yhatTrain);
        errTest = computeClassificationError(trainLabels(testId,:), yhatTest);
        %Store 
        trainErrors(i, ft) = errTrain;
        testErrors(i, ft) = errTest;
    end
end

%Average errors (+std)
meanTrainErrors = mean(trainErrors)
meanTestErrors = mean(testErrors)

stdTrainErrors = std(trainErrors);
stdTestErrors = std(testErrors);

%Let's plot this
%Train
figure
plot(1:maxFt, meanTrainErrors, 'LineWidth', 2, 'Color', 'r'); hold on
xlabel('Number of features');
ylabel('Mean training error'); 

%Test
plot(1:maxFt, meanTestErrors, 'LineWidth', 2, 'Color', 'b'); 
xlabel('Number of features');
ylabel('Mean testing error'); hold off


