function [ error] = trainTestError( trainSet, trainLabels, testSet, testLabels, model, ratio )
% Given a model trained on training dataset, computes train and test class
% errors with specified ratio. Real data classification sets, trainLabels
% and testLabels are necessary to compute errors.
% Output is of format error=[trainError,testError]. 
trainPrediction = predict(model, trainSet);
testPrediction = predict(model,testSet);

trainError = computeClassError(trainLabels, trainPrediction, ratio);
testError = computeClassError(testLabels, testPrediction, ratio);

error=[trainError, testError];
end

