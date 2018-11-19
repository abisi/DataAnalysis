%% Guidesheet 7 : PCA and Regression

clear all;
close all;
load('../data/Data.mat');

%% Data set partitioning and PCA 

%Partition
proportion = 0.7;     
rows = size(Data,1);    
train = Data(1:round(rows*proportion),:);
test = Data(rows-round(rows*(1-proportion)):end,:);

%PCA

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
chosen_PCs = 579; %TO FILL IN
target_posx = PosX(train); %y
target_posy = PosY(train);

FMx = pca_train(target_posx,1:chosen_PCs); 
FMy = pca_train(target_posy,1:chosen_PCs); 

I = ones(size(target_posx,1),1);

X_posx = [Ix FM];
X_posy = [Iy FM];

bx = regress(target_posx,X_posx(train,1:chosen_PCs)); %b: coefficient
by = regress(target_posy,X_posy(train,1:chosen_PCs));

%Mean-square error calculation
mse_posx = immse(target_posx, X_posx * bx);
mse_posy = immse(target_posy, X_posy * by);

%Plot real vectors and regressed ones
plot(ms)

%% Regression - 2nd order polynomial regressor
X_posx_order2 = [Ix feature_matrix feature_matrix.^2];
X_posy_order2 = [Iy feature_matrix feature_matrix.^2];

