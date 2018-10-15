function [error, accur] = classificationError(labels, prediction)
%CLASSIFICATIONERROR Computes classification error and accuracy.
%   Classif. accuracy :  number of correctly classified samples
%   (regardless of class) over number of total samples.
%   Classif. error : number of missclasssified samples over number of total
%   samples, or, alternatively, [1 - (classif. accuracy)].

counter = 0;
for i=1:length(prediction)
    if prediction(i,1) == labels(i,1)
        counter = counter + 1;
    end
end

accur = counter / length(prediction);
error = 1 - accur;
end

