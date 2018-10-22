function [error] = trainTestError(trainSet, trainLabels, testSet, testLabels, model, ratio )
% TRAINTESTERROR Both predicts and computes class error using given
%                training set et test set.
% 	Given a model trained on training dataset, computes train and test class
%   errors with specified ratio. Real data classification sets, trainLabels
%   and testLabels are necessary to compute errors.
%   INTUTS:
%       trainSet: training set with features
%       trainLabels: training labels
%       testSet: test set with features
%       testLabels: test labels (often not available)
%       model: classifier model (fitcdiscr)
%       ratio: ratio used for class error
%   OUTPUT:  ERROR=[TRAINERROR, TESTERROR]. 

trainPrediction = predict(model, trainSet);
testPrediction = predict(model,testSet);

trainError = computeClassError(trainLabels, trainPrediction, ratio);
testError = computeClassError(testLabels, testPrediction, ratio);

error=[trainError, testError];
end

