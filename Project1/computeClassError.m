function [classError] = computeClassError(labels, prediction, ratio)
%CLASSERROR Computes class error.
%   For unbalanced classes A and B across a dataset, the class error is defined in
%   order to balance the influence of errors in class A and class B.
%   labels : true target labels
%   prediction : predicted classes
%   ratio : number between 0 and 1 for the proportion of CLASS A (class
%   Correct), then for second factor : [1 - ratio]

% Error counters:
correctClassError = 0;
errorClassError = 0;

for i=1:length(prediction)
    if(prediction(i)~=labels(i,1) && labels(i,1) == 0)
        correctClassError = correctClassError + 1;
    else if (prediction(i)~=labels(i,1) && labels(i,1)== 1)
            errorClassError = errorClassError + 1;
        end
    end
end

nErr=nnz(labels); %because 1 corresponds to error class
nCorr=length(labels)-nErr;

classError = ratio .* (correctClassError/nCorr) + (1-ratio) .* (errorClassError/nErr);
end



