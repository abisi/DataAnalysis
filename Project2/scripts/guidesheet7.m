%% Guidesheet 7 : PCA and Regression

clear all;
close all;
load('../data/data.mat');

%% Data set partitioning and PCA 

%Partition
data = cumsum(ones(size(Data)));  % some test data
proportion = 0.7;     
rows = size(data,1);  
idx_vector = false(rows,1);    
idx_vector(1:round(proportion*rows)) = true;   
idx_vector = idx_vector(randperm(rows));   % randomise order
train = data(idx_vector,:);
test = data(~idx_vector,:);

%PCA

[std_train, mu, sigma] = zscore(train);
std_test = (test - mu ) ./ sigma; 

[coeff, score, latent] = pca(std_train);
pca_data = std_train * coeff;

%Choose PCs (+ graph)


%Regression

%chosen_PCs = ...; TO FILL IN
target_posx = PosX(train);
target_posy = PosY(train);

b = regress(target_posx(train),data(train,chosen_PCs))

