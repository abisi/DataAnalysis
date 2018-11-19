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

pca_test = std_test * coeff;

%Choose PCs (+ graph)


%Regression - linear
%chosen_PCs = ...; %TO FILL IN
target_posx = PosX(train); %y
target_posy = PosY(train);

%feature_matrix = ...; %TO FILL IN
I = ones(size(target_posx(train),1),1);
%For the x-data vector
X_posx = [I feature_matrix];

bx = regress(target_posx(train),X_posx(train,chosen_PCs)); %b: coefficient

%For the y-data vector
X_posy = [I feature_matrix];

by = regress(target_posy(train),X_posy(train,chosen_PCs));

%Mean-square error calculation
mse_posx = immse(target_posx, X_posx * bx);
mse_posy = immse(target_posy, X_posy * by);


%Plot real vectors and regressed ones


%Regression - 2nd order polynomial regressor
X_posx_order2 = [Ix feature_matrix feature_matrix.^2];
X_posy_order2 = [Iy feature_matrix feature_matrix.^2];

