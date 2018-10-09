%guidesheet 1 : data exploration

%closes previously open figures
close all

%Creation of a categorical array from trainLabels. Constructor takes binary
%values from original array and map them to error and correct categorical
%values.
C = categorical(trainLabels,[1 0],{'Error','Correct'});

%Generate histogram of trainLabels to visually compare frequency of label
%apparaition to theoretical incidence rates : p(error)= 30% , p(correct)=
%70%.
histogram(trainLabels,'Normalization','probability', 'BinWidth',0.5);
figure;

%Examining data distribution for feature 650 in samples with classification
%'correct'. Same thing for samples with classification 'incorrect'.
%Distributions are superposed to examine whether or not they are similar.
%Similar distributions mean that the feature is not good for establishing
%the classification. Highly disting distributions mean the feature is good
%for establishing the classification.
histogram(trainData(trainLabels == 0,650),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,650),50, 'Normalization', 'probability'); hold off
figure;

%Same thing, comparing data distribution @ feature 712 with respect to
%classes.
histogram(trainData(trainLabels == 0,712),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,712),50, 'Normalization', 'probability'); hold off
figure;

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
boxplot(trainData(:,650),groupingMatrix);
%setting x-axis labels
set(gca,'XTickLabel',{'Correct','Error'});

%boxplot of data distribution @feature 720 with respect to classes
figure
boxplot(trainData(:,720), groupingMatrix); 
set(gca,'XTickLabel',{'Correct','Error'});

%% boxplots with confidence intervals

%we see that confidence intervals for respective means overlap. We conclude
%that classes display similar data distributions @feature 650. One should
%not try and establish the classification based on this feature.
figure
boxplot(trainData(:,650),groupingMatrix, 'Notch', 'on'); 

%confidence intervals for respective means appear distinct. We suspect that
%classes display significantly different data distributions @feature 720.
%If significant, this difference means that feature 720 can be used to
%establish the desired classification.
%Note that we still need to check statistical significance of mean
%difference between classes @ feature 720 using t-testing.
figure
boxplot(trainData(:,720), groupingMatrix, 'Notch', 'on'); 

%% t-tests

