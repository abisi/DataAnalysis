histogram(trainData(:,500), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5); hold on;
histogram(trainData(:,1500), 50, 'FaceColor', 'y', 'FaceAlpha', 0.5);

C = categorical(trainLabels',[1 0],{'Error','Correct'});
figure;
histogram(C,'Normalization','probability', 'BarWidth',0.5);