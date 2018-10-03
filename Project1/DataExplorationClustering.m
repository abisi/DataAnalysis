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
% A reason : exactly same algo performed with exactly same parameters

%% Let's try for different values of k
k_alt = [2,3,4,5,6];
meanPointToCentroid = zeros(1,5);
for i = 1:length(k_alt)
    [clusterIndexes, centroids, sumD] = kmeans(spikes,k_alt(i));
    % Say we form 3 clusters. sumD will be a 3x1 vector. Each line is the
    % distance corresponding to the sum of the squared euclidean distance 
    % between each cluster point and centroid.
    % Likewise, centroids will be a 3 x 100 matrix. Each line represents a
    % cluster and the 100 values are the clusters centroid values (we need
    % a centroid value at each feature).
    meanPointToCentroid(i) = sum(sumD)/numberPeaks;
    % We calculated the average point to centroid distance (irrespective
    % of which cluster appartenance)
    figure
    gplotmatrix(spikes(:,selectedFeatures),[],clusterIndexes); 
end

figure
plot(k_alt, meanPointToCentroid);

% Increasing number of clusters reduces the mean point to centroid
% distance. However using gplotmatrix to compare spikes at selected
% features shows that adding too many new clusters causes overlapping with
% pre-existing ones. -> The mean distance metric alone is not sufficient to
% determine number of clusters.
% According to us, the optimal number of clusters is 3.

%% Use Matlab function evalclusters to determine optimal number of clusters 
% evalclusters finds optimal number of clusters based on an internal
% criterion. 

EVA = evalclusters(spikes, 'kmeans','CalinskiHarabasz','KList',[2:6]);
EVA2 = evalclusters(spikes, 'kmeans','DaviesBouldin','KList',[2:6]);
% The Davies-Bouldin Index evaluates intra-cluster similarity and
% inter-cluster differences.

% These evaluation methods indicate that 2 clusters generate the best
% results. We see again that internal metrics and statistics do not always
% yield the best interpretation of data.
%% evalclustering methods that don't finish -> dataset too large
EVA3 = evalclusters(spikes, 'kmeans','gap','KList'[2:6]);
% The gap statistic compares the total within intracluster variation for
% different values of k with their expected values under null reference
% distribution of the data, i.e. a distribution with no obvious clustering
EVA4 = evalclusters(spikes, 'kmeans','silhouette','KList',[2:6]);
% The Silhouette Index measures the distance between each data point,
% the centroid of the cluster it was assigned to and the closest centroid
% belonging to another cluster.

