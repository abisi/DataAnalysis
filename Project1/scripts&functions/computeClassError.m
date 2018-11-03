function [error, accur] = computeClassError(labels, prediction)
%CLASSERROR Computes class error.
%   For unbalanced classes A and B across a dataset, the class error is defined in
%   order to balance the influence of errors in class A and class B.
%   labels : true target labels
%   prediction : predicted classes
%


% Ratio of correct '0'
ratio = length((find(labels == 0))) / length(labels);
% Error counters:
correctClassError = 0;
errorClassError = 0;

for i=1:length(labels)
    if(prediction(i) ~= labels(i) && labels(i) == 0)
        correctClassError = correctClassError + 1;
    else if (prediction(i) ~= labels(i) && labels(i) == 1)
            errorClassError = errorClassError + 1;
        end
    end
end

nErr=nnz(labels); %because 1 corresponds to error class
nCorr=length(labels)-nErr;

error = ratio * (correctClassError/nCorr) + (1-ratio) * (errorClassError/nErr);
accur = 1 - error;

end



