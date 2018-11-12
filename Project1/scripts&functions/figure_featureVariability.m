%% PART 1 : Statistical significance
%% Feature Variability figure
close all;
clear all;
load('../data/trainSet.mat');
load('../data/trainLabels.mat');
figure

%colors
c1=[0.4 0.8 0.7];
c2=[1 0.7 0.5];
c3=[0.9 0 0.2];

%Class histograms for feature 650 with threshold
subplot(2,2,1);
h1=histogram(trainData(trainLabels == 0,650),50, 'Normalization', 'probability'); hold on
h2=histogram(trainData(trainLabels == 1,650),50, 'Normalization', 'probability');
set(h1,'FaceColor',c1);
set(h2,'FaceColor',c2);
y=ylim;
threshold=line([0.505 0.505], [y(1) y(2)]);
set(threshold,'LineWidth',3,'color',c3);
legend('Correct', 'Error', 'Threshold');
xlabel('Signal amplitude');
ylabel('Normalized proportion');
title('Normalized distribution of correct and error related ERPs (feature 650)');

%Class histograms for feature 712 with threshold
subplot(2,2,2);
h1=histogram(trainData(trainLabels == 0,712),50, 'Normalization', 'probability'); hold on
h2=histogram(trainData(trainLabels == 1,712),50, 'Normalization', 'probability');
set(h1,'FaceColor',c1);
set(h2,'FaceColor',c2);
y=ylim;
threshold=line([0.64 0.64], [y(1) y(2)]);
set(threshold,'LineWidth',3,'color',c3);
legend('Correct', 'Error', 'Threshold');
xlabel('Signal amplitude');
ylabel('Normalized proportion');
title('Normalized distribution of correct and error related ERPs (feature 712)');

%Class histograms help us to optimally place threshold because we can
%minimize false classifications by placing threshold at intersection of
%conditional probability distributions.

%Creating grouping matrix to differentiate between class Error and class
%Correct data when boxplotting.
CorrectTag = ones([nnz(trainLabels==0) 1]);
ErrorTag = 2 * ones([nnz(trainLabels==1) 1]);
groupingMatrix = [CorrectTag; ErrorTag];

%we see that confidence intervals for respective means overlap. We conclude
%that classes display similar data distributions @feature 650. One should
%not try and establish the classification based on this feature.
subplot(2,2,3);
boxplot(trainData(:,650),groupingMatrix, 'Notch', 'on','Colors','k');
set(findobj(gcf,'Tag','Median'),'Color',c3);
set(findobj(gcf,'Tag','Outliers'),'MarkerEdgeColor','k');
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 650 boxplot for correct and error related ERPs');

%confidence intervals for respective means appear distinct. We suspect that
%classes display significantly different data distributions @feature 712.
subplot(2,2,4);
boxplot(trainData(:,712), groupingMatrix, 'Notch', 'on','Colors','k');
set(findobj(gcf,'Tag','Median'),'Color',c3);
set(findobj(gcf,'Tag','Outliers'),'MarkerEdgeColor','k');
set(gca,'XTickLabel',{'Correct','Error'});
title('Feature 712 boxplot for correct and error related ERPs');





