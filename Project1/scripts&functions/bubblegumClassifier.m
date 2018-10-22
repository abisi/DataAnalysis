function [predictionVector] = bubblegumClassifier(trainingData, trainingLabels, testingData, ratio )
%Mutidimensional model; ratio is important because it defines how optimal
%thresholds are calculated (cf. class error formula)

%Determining optimal threashold for each feature
determinedThresholds=[];

for i=1:length(trainingData)
    thresh = computeOptimalThreashold( [0:0.01:1],ratio,i, trainingData, trainingLabels );
    determinedThresholds=[determinedThresholds thresh];
end

%Predicting
predicted=[];
numberSamples=size(testingData,1);

for i=1:numberSamples
     newPrediction = mrPredictor( determinedThresholds, testingData(i,:));
     predicted=[predicted newPrediction];
end

predictionVector=predicted;

end

