% %size of feature vector
% sizeFeatureVector=length(spikes(1,:));
% %number of samples
% numberPeaks=length(spikes(:,1));
% %plot de tous les peaks
% plot(spikes(1000:1100,:));
% %choisir 2 pts et visualiser
% plot(spikes(:,20));
% figure;
% plot(spikes(:,60));
% figure;
% %histogram des 2 features
% hist(spikes(:,20),100);
% figure;
% hist(spikes(:,60),100);
% figure;
% %boxplot des 2 features
% boxplot(spikes(:,20));
% figure;
% boxplot(spikes(:,60));
% figure;
% %on voit que l'histogram nous fournit les info sur le type de distrubution
% %et feature seperatibilty tandis que le boxlot nous informe sur la moyenne
% %et les outliers
% plotmatrix(spikes(:,1:5:end));
% figure;
% scatter(spikes(:,41),spikes(:,61));
% figure;
% scatter3(spikes(:,41),spikes(:,61),spikes(:,56));
%we decided on the following features
selectedFeatures= [41 56 61];
%kmeans hands on
k=3;
clusterIndexes=kmeans(spikes,k);
gplotmatrix(spikes(:,selectedFeatures),[],clusterIndexes);
%plot spike profiles (mean of each cluster)
cluster1=[];
counter1=0;
cluster2=[];
counter2=0;
cluster3=[];
counter3=0;
for i=1:6000
    if clusterIndexes(i)==1
        cluster1=[cluster1 spikes(i,:)];
        counter1=counter1+1;
    elseif clusterIndexes(i)==2
        cluster2=[cluster2 spikes(i,:)];
        counter2=counter2+1;
    elseif clusterIndexes(i)==3
        cluster3=[cluster3 spikes(i,:)];
        counter3=counter3+1;
    end
end
cluster1=cluster1/counter1;
cluster2=cluster2/counter2;
cluster3=cluster3/counter3;
% plot([1:1:100],cluster1);
% hold on;
% plot([1:1:100],cluster2);
% plot([1:1:100],cluster3);
% figure;