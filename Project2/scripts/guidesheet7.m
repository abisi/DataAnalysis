%% Guidesheet 7 : PCA and Regression

clear all;
close all;
load('../data/Data.mat');

%% Data set partitioning and PCA 

%Partition
proportion = 0.7;     
rows = size(Data,1);    
train = Data(1:round(rows*proportion),:);
test = Data(rows-round(rows*(1-proportion)):end,:); %here we keep order because we wanna predict future values based on past values

[std_train, mu, sigma] = zscore(train);
std_test = (test - mu ) ./ sigma; 

%PCA

[coeff, score, latent] = pca(std_train);
pca_train = std_train * coeff;
pca_test = std_test * coeff;

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
chosen_PCs = 741; %TO FILL IN
target_posx = PosX(1:round(rows*proportion)); %y
target_posy = PosY(1:round(rows*proportion));

FM = pca_train(:,1:chosen_PCs); 
I = ones(size(target_posx,1),1);
X = [I FM];

%Regression
bx = regress(target_posx,X(1:round(rows*proportion),1:chosen_PCs)); %b: coefficient
by = regress(target_posy,X(1:round(rows*proportion),1:chosen_PCs));

%Mean-square error calculation
mse_posx = immse(target_posx, X(:,1:chosen_PCs) * bx);
mse_posy = immse(target_posy, X(:,1:chosen_PCs) * by);

%% Plot
figure
plot(1:rows, PosX); hold on
plot(1:rows, PosY);

%% Regression - 2nd order polynomial regressor
X_posx_order2 = [Ix feature_matrix feature_matrix.^2];
X_posy_order2 = [Iy feature_matrix feature_matrix.^2];

