function [predictionVector]=computePrediction(dataSet, featureNumber, thresholdValue)
%COMPUTEPREDICTION compute the prediction vector of a specific featureNumber belonging to a data set. 
%It compares the value of the data set with given threshold value.

predictionVector=[]; %Be careful, the predictionVector will be linear!
for i=1:length(dataSet(:,1))
    if(dataSet(i,featureNumber)>thresholdValue)
        predictionVector(i)=1;
    else
        predictionVector(i)=0;
    end
end

end