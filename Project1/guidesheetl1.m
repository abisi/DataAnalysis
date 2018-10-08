%guidesheet 1 : data exploration

close all
%statistical significance 

C = categorical(trainLabels',[1 0],{'Error','Correct'});
figure;
histogram(C,'Normalization','probability', 'BarWidth',0.5);


figure  %similar
histogram(trainData(trainLabels == 0,650),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,650),50, 'Normalization', 'probability'); hold off

figure %different
histogram(trainData(trainLabels == 0,712),50, 'Normalization', 'probability'); hold on
histogram(trainData(trainLabels == 1,712),50, 'Normalization', 'probability'); hold off

%% boxplot

close all 

figure
boxplot([trainData(trainLabels == 0,650), trainData(trainLabels == 1,650)]); 

figure
boxplot([trainData(trainLabels == 0,720), trainData(trainLabels == 1,720)]); 


%% boxplot with confidence interval

figure
boxplot([trainData(trainLabels == 0,650), trainData(trainLabels == 1,650)], 'Notch', 'on'); 

figure
boxplot([trainData(trainLabels == 0,720), trainData(trainLabels == 1,720)], 'Notch', 'on'); 

