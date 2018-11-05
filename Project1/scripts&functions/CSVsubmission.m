function [] = CSVsubmission(prediction, filename, folder);
%SVSUBMISSION Predicts and creates .csv submission based on the label-less test set.
%   prediction : prediction vector for the test set ('0' or '1')
%   filename : name of the .csv file to be made 
%   folder : location of the .csv file

load('../data/testSet.mat');
%Predict
prediction = 
%To CSV
labelToCSV(prediction, filename, folder);



