%% Guidesheet3: Model selection and nested cross-validation

%%Cross-validation for hyperparameter optimization

%Fisher-based feature selection
k = 10; %number of folds
maxFt = 20; %max number of features
cp_labels = cvpartition(trainLabels, 'kfold', k);%partition
trainErrors = zeros(k, maxFt); %initialization
testErrors = zeros(k, maxFt);

for ft = 1:maxFt %incrementing number of features
    for i=1:cp_labels.NumTestSets
        trainId = cp_labels.training(i); %markers for training
        testId = cp_labels.test(i); %markers for testing
        %Pick ft-th best features
    	[orderedInd, orderedPower] = rankfeat(trainData(trainId,:), trainLabels(trainId,:), 'fisher');
        bestInd = orderedInd(1:ft); %selection of best n features
        %Build model and predict
        classifierDiagLin = fitcdiscr(trainData(trainId, bestInd), trainLabels(trainId,:), 'DiscrimType', 'diaglinear');
        yhatTrain = predict(classifierDiagLin, trainData(trainId, bestInd));
        yhatTest = predict(classifierDiagLin, trainData(testId, bestInd));
        %Compute both errors
        errTrain = computeClassificationError(trainLabels(trainId,:), yhatTrain);
        errTest = computeClassificationError(trainLabels(testId,:), yhatTest);
        %Store 
        trainErrors(i, ft) = errTrain; % Each column respresents a different model (different amount of features used)
        testErrors(i, ft) = errTest; % Each row of selected column contains different fold respective errors.
    end
end

%Average errors (+std)
meanTrainErrors = mean(trainErrors) %mean of a matrix returns row vector containing mean value of each column
meanTestErrors = mean(testErrors)

stdTrainErrors = std(trainErrors); % returns std dev of each column in a row vector
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
%% Nested cross-validation for performance estimation



