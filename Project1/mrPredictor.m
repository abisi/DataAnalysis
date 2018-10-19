function [ bestPrediction ] = mrPredictor( thresholds, sample )
%Given thresholds and a given sample, Mr.Predictor calculates a prediction
%based on each feature. Then Mr.Predictor identifies the most frequent
%prediction and returns it.
predVals=[];

for i=1:length(sample)
    if(sample(i)>thresholds(i))
        predVals=[predVals 1];
    else
        predVals=[predVals 0];
    end
end

classOneFreq=nnz(predVals)/length(predVals);

if(classOneFreq>0.5)
    bestPrediction=1;
else
    bestPrediction=0;
end


end

