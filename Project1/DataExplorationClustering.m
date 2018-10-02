% Data exploration and Clustering - Guidesheet 0

%% Data exploration
% Size of feature vector
sizeFeatureVector = length(spikes(1,:));
% Number of samples
numberPeaks = length(spikes(:,1));
% Plot de tous les samples / ou quelqu'uns (ici 100) -> pas bon pour
% compter le #Neurons
figure
plot(spikes(1000:1100,:));
% Choisir 2 features et visualiser
figure
plot(spikes(:,20)); hold on
plot(spikes(:,60)); hold off
% Histogram des 2 features
figure
hist(spikes(:,20),100); hold on
hist(spikes(:,60),100); hold off
% Boxplot des 2 features
figure
boxplot(spikes(:,20));
figure
boxplot(spikes(:,60)); hold off
% On voit que l'histogramme nous fournit les infos sur le type de distribution
% et "feature seperatibility" tandis que le boxplot nous informe sur la moyenne
% et les valeurs aberrantes (outliers)

% Spikes sur sous-ensembles des features & 3D scatterplot
figure
plotmatrix(spikes(:,1:5:end));
figure
scatter(spikes(:,41),spikes(:,61));
figure;
scatter3(spikes(:,41),spikes(:,61),spikes(:,56));

% We decided on the following features looking at plotmatrix & scatter3
selectedFeatures= [41 56 61];

%% K-Means clustering

k=3; %as assumed from previously

clusterIndexes = kmeans(spikes,k);
figure
gplotmatrix(spikes(:,selectedFeatures),[],clusterIndexes);

% Plot spike profiles (mean of each cluster)
cluster1 = zeros(1, sizeFeatureVector);
counter1 = 0;
cluster2 = zeros(1, sizeFeatureVector);;
counter2 = 0;
cluster3 = zeros(1, sizeFeatureVector);;
counter3 = 0;
for i = 1:numberPeaks
    if clusterIndexes(i) == 1
        cluster1 = cluster1 + spikes(i,:);
        counter1 = counter1+1;
    elseif clusterIndexes(i) == 2
        cluster2 = cluster2 + spikes(i,:);
        counter2 = counter2+1;
    elseif clusterIndexes(i) == 3
        cluster3 = cluster3 + spikes(i,:);
        counter3 = counter3+1;
    end
end

% Neuron profiles (means)
cluster1 = cluster1 / counter1;
cluster2 = cluster2 / counter2;
cluster3 = cluster3 / counter3;

%Let's plot the profiles
figure
x_t = [1:sizeFeatureVector];
plot(x_t,cluster1); hold on
plot(x_t,cluster2);
plot(x_t,cluster3); hold off 
% We observe that the profiles are quite different yet could be more differentiated perhaps 
% yet different enough to justify three diff. neurons


%% Hands-on c'd

%Repeating the clustering :
clusterIndexes2 = kmeans(spikes,k);
figure
gplotmatrix(spikes(:,selectedFeatures),[],clusterIndexes2);
% We observe that colored clusters do not look diff.
% A reason : ?

%% Let's try for different values of k
k_alt = [3,4,5,6];
meanSums = zeros(1,4);
for i = 1:k_alt
    [clusterIndexes, centroids, sumD] = kmeans(spikes,k_alt(i));
    meanSums(i) = mean(sumD);
    figure
    gplotmatrix(spikes(:,selectedFeatures),[],clusterIndexes); 
end

figure
plot(k_alt, meanSums);

% On average the mean of the sums decreases. Yet with increasing number of clusters, sumD does not allow us to see
% differences (as it only gives the sum of distances per clusters?) because
% then the clusters are clearly wrong.

%% Evaluate clustering algorithms and optimal number of 
% clusters evalclusters based on an internal criterion :

%EVA = evalclusters(spikes, 'kmeans')