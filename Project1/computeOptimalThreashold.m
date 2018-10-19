function [ bestThreshold, minimizedClassError ] = computeOptimalThreashold( rangeVector,ratio,featureIndex, dataSet, labels )
%Computes optimal threashold (minimizes classError) at given feature

%Calculate the class error
thresholdValues=rangeVector;
classErrorVector=[];
classificationErrorVector=[];

for t=1:length(thresholdValues)
    predVector=computePrediction(dataSet, featureIndex, thresholdValues(t)); %return a linear vector
    classError = computeClassError(labels, predVector, ratio);  
    classErrorVector=[classErrorVector classError];
    classificationError = computeClassificationError(labels, predVector'); %computeClassificationError takes a cloumn vector as argument
    classificationErrorVector=[classificationErrorVector classificationError];
end

[classErrorMin, indexMin]=min(classErrorVector)
bestThreshold=thresholdValues(indexMin)
minimizedClassError=classErrorMin;


end