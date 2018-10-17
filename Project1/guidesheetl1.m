%% PART 1 : Statistical significance
%% Histograms
%closes previously open figures
close all

%Creation of a categorical array from trainLabels. Constructor takes binary
%values from original array and map them to error and correct categorical
%values.
C = categorical(trainLabels',[1 0],{'Error','Correct'});

%Generate histogram of trainLabels to visually compare frequency of label
%apparaition to theoretical incidence rates : p(error)= 30% , p(correct)=
%70%.
histogram(C,'Normalization','probability', 'BarWidth',0.5);
ylabel('Percentage');
title('Proportion of correct- and error-related ERPs');

%Examining data distribution for feature 650 in samples with classification
%'correct'. Same thing for samples with classification 'incorrect'.
%Distributions are superposed to examine whether or not they are similar.
%Similar distributions mean that the feature is not good for establishing
%the classification. Highly disting distributions mean the feature is good
%for establishing the classification.
figure;
histogram(trainData(trainLabels == 0,650),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,650),50, 'Normalization', 'probability'); hold off
legend('Correct', 'Error');
xlabel('Signal amplitude');
ylabel('Normalized proportion');
title('Normalized distribution of correct and error (feature 650)');

%Same thing, comparing data distribution @ feature 712 with respect to
%classes.
figure;
histogram(trainData(trainLabels == 0,712),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,712),50, 'Normalization', 'probability'); hold off
legend('Correct', 'Error');
xlabel('Signal amplitude');
ylabel('Normalized proportion');
title('Normalized distribution of correct- and error-related ERPs (feature 712)');

%% boxplot

%getting rid of previous figures
close all 

%Boxplots of class data @feature 650. Class data is constituted by all
%respective sample values @ feature 650. 

%Creating grouping matrix to differentiate between class Error and class
%Correct data when boxplotting.
CorrectTag = ones([nnz(trainLabels==0) 1]);
ErrorTag = 2 * ones([nnz(trainLabels==1) 1]);
groupingMatrix = [CorrectTag; ErrorTag];

%boxplot of data distribution @feature 650 with respect to classes
figure
boxplot(trainData(:,650),groupingMatrix);
%setting x-axis labels
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 650 boxplot for correct- and error-related ERPs');

%boxplot of data distribution @feature 712 with respect to classes
figure
boxplot(trainData(:,712), groupingMatrix); 
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 712 boxplot for correct- and error-related ERPs');

%% boxplots with confidence intervals

%we see that confidence intervals for respective means overlap. We conclude
%that classes display similar data distributions @feature 650. One should
%not try and establish the classification based on this feature.
figure
boxplot(trainData(:,650),groupingMatrix, 'Notch', 'on'); 
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 612 boxplot for correct- and error-related ERPs');

%confidence intervals for respective means appear distinct. We suspect that
%classes display significantly different data distributions @feature 712.
%If significant, this difference means that feature 712 can be used to
%establish the desired classification.
%Note that we still need to check statistical significance of mean
%difference between classes @ feature 712 using t-testing.
figure
boxplot(trainData(:,712), groupingMatrix, 'Notch', 'on');
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 712 boxplot for correct- and error-related ERPs');

%% t-tests

%Sets of Data to be compared
CorrectClassDataF712=trainData(trainLabels==0,712);
ErrorClassDataF712=trainData(trainLabels==1,712);

CorrectClassDataF650=trainData(trainLabels==0,650);
ErrorClassDataF650=trainData(trainLabels==1,650);

%We use ttest2 because data sets under cannot be paired (more data points
%in correctClassData than ErrorClassData).
%T-testing confirms what we suspected when looking at confidence interval
%boxplots.

%Null hypothesis (average of both data sets is equal) can be
%rejected @f712 at 5% significance lvl. It cannot be rejected for f650.

%p-value (probability of observing the given result, or one more extreme,
%by chance if the null hypothesis is true) for f712 is extermely low
%(3.8e-16). For f650, the p-value is very large (0.4).

%One cannot simply use t-test for any feature. First we need to check that
%data from respective classes is normally distributed at this feature and
%that their variance is similar (histogram superposition).

[H_f650,P_f650]=ttest2(CorrectClassDataF650,ErrorClassDataF650);
[H_f712,P_f712]=ttest2(CorrectClassDataF712,ErrorClassDataF712);
%% PART 2 : Feature Thresholding
%% Determining optimal theshold through histograms
close all;
clear all;
load('trainSet.mat');
load('trainLabels.mat');

%Class histograms for feature 712 with threshold
figure;
histogram(trainData(trainLabels == 0,712),100, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,712),100, 'Normalization', 'probability');
threshold=line([0.64 0.64], [0 0.04]);
set(threshold,'LineWidth',4,'color','red');

%Class histograms for feature 720 with threshold
figure
histogram(trainData(trainLabels == 0,720),100, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,720),100, 'Normalization', 'probability');
threshold=line([0.47 0.47], [0 0.05]);
set(threshold,'LineWidth',4,'color','red');

%Class histograms help us to optimally place threshold because we can
%minimize false classifications by placing threshold at intersection of
%conditional probability distributions.
%% Plot of datapoints of two features with optimal threashold as a line
figure
correct712=plot(trainData(trainLabels==0,712));hold on
error712=plot(trainData(trainLabels==1,712));
correct720=plot(trainData(trainLabels==0,720));
error720=plot(trainData(trainLabels==1,720));
set(correct712,'color','blue');
set(error712,'color','green');
set(correct720,'color','yellow');
%set(error720,'color','purple');
%-> Cannot set color here because pixels color is already set previously?
threshold=line([0 500], [0.64 0.64]);
set(threshold,'LineWidth',4,'color','red');

%We see that for feature 712 (feature used to define threshold), correct
%class (blue)is predominantly below our threashold whereas error class
%(green) is predominantly above.

%Unfortunately, we see that data from other features such as feature 720
%does not obey our established threashold. We see that both error class
%(purple) and correct class (yellow) lie predominantly below the threashold.

% We conclude that the shortcoming of feature thresholding is that we
% cannot establish a universal threashold that holds for any feature within
% our data.
%% classification error
close all;
clear all;
load('trainSet.mat');
load('trainLabels.mat');

threshVal=0.64;
%predicting
predicted=[];
for i=1:597
    if(trainData(i,712)>0.64)
        predicted(i)=1;
    else
        predicted(i)=0;
    end
end
%comparing predicted to labels
correctCounter=0;
for i=1:597
    if(predicted(1,i)==trainLabels(i,1))
        correctCounter=correctCounter+1;
    end
end
%classification accuracy
classificationAccuracy=correctCounter/597;
%classification error
classificationError=1-classificationAccuracy;
%error typing
corrError=0;
errError=0;
for i=1:597
    if(predicted(1,i)~=trainLabels(i,1) && trainLabels(i,1)==0)
        corrError=corrError+1;
    else if (predicted(1,i)~=trainLabels(i,1) && trainLabels(i,1)==1)
            errError=errError+1;
        end
    end
end
%classError
numberErr=nnz(trainLabels);
numberCorr=597-numberErr;
classError=0.5*(errError/numberErr)+0.5*(corrError/numberCorr);
classError2=0.33*(errError/numberErr)+0.66*(corrError/numberCorr);


%cleanup
clear correctCounter corrError errError i numberCorr numberErr predicted;

%% Sclaing thresholding method to multiple dimensions
%We can scale up method by judging data entry at 2 features with distinct
%thresholds.
%% scatterplot
err=scatter(trainData(trainLabels==0,712),trainData(trainLabels==0,720)); hold on;
corr=scatter(trainData(trainLabels==1,712),trainData(trainLabels==1,720));

%% Plot class error and classification error as a function of threshold values
%Calculate the class error
thresholdValues=[0.4:0.05:0.8];
ratio=0.5;
classErrorVector=[];
classificationErrorVector=[];

for t=1:length(thresholdValues)
    predVector=computePrediction(trainData, 712, thresholdValues(t)); %return a linear vector
    classError = computeClassError(trainLabels, predVector, ratio);  
    classErrorVector=[classErrorVector classError];
    classificationError = computeClassificationError(trainLabels, predVector'); %computeClassificationError takes a cloumn vector as argument
    classificationErrorVector=[classificationErrorVector classificationError];
end

figure;
scatter(thresholdValues, classErrorVector, 10, 'r', 'filled'); 
xlabel('Threshold');
ylabel('Class error');
title('Class error as a function of thresold values');

figure;
scatter(thresholdValues, classificationErrorVector, 10, 'b', 'filled');
xlabel('Threshold');
ylabel('Classification error');
title('Classification error as a function of threshold values');

