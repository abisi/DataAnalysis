%% Guidesheet 7 : PCA and Regression

clear all;
close all;
load('../data/data.mat');

%% Data set partitioning and PCA 

%Partition
proportion = 0.7;     
rows = size(Data,1);  
idx_vector = false(rows,1);    
idx_vector(1:round(proportion*rows)) = true;   
idx_vector = idx_vector(randperm(rows));   % randomise order
train = Data(idx_vector,:);
test = Data(~idx_vector,:);

%PCA

%[std_train, mu, sigma] = zscore(train); NO because time-correlated
%std_test = (test - mu ) ./ sigma; 

[coeff, score, latent] = pca(train);
pca_train = train * coeff;
pca_test = test * coeff;

%Choose PCs (+ graphs)
cumVar=cumsum(latent)/sum(latent);
numPC=1:length(latent);
figure;
plot(numPC, cumVar, 'r'); hold on;
xlabel('Number of PCs');
ylabel('Percentage of the total variance');
title('Total information carried by Principal Components');

idx90=find(cumVar>0.9);
pc90=numPC(idx90(1));
threshold90=line([pc90 pc90], [0 1]);
set(threshold90,'LineWidth',2,'color','blue');

figure
bar(latent);

%% Regression - linear
chosen_PCs = 578; %TO FILL IN
target_posx = PosX(train); %y
target_posy = PosY(train);

FM = pca_train(:,1:chosen_PCs); 

Ix = ones(size(target_posx,1),1);
Iy = ones(size(target_posy,1),1);
X_posx = [Ix FM];
X_posy = [Iy FM];

bx = regress(target_posx,X_posx(train,1:chosen_PCs)); %b: coefficient
by = regress(target_posy,X_posy(train,1:chosen_PCs));

%Mean-square error calculation
mse_posx = immse(target_posx, X_posx * bx);
mse_posy = immse(target_posy, X_posy * by);

%Plot real vectors and regressed ones


%% Regression - 2nd order polynomial regressor
X_posx_order2 = [Ix feature_matrix feature_matrix.^2];
X_posy_order2 = [Iy feature_matrix feature_matrix.^2];

