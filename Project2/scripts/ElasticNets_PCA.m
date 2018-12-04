%% Lasso VS Elastic Nets
%% Data loading
%The aim here is to distinguish the difference between Lasso and EN for a
%fixed value of alpha. With PCA.
clear all;
close all;
load('../data/Data.mat');

%% Data partitionning
train_proportion = 0.3; %use 5% of the dataset for training
validation_proportion = 0.4;
rows = size(Data,1);

sep_idx_train = ceil(rows*train_proportion);
sep_idx_val = ceil(rows*(train_proportion + validation_proportion));

%Train targets
pos_x_train = PosX(1:sep_idx_train);
pos_y_train = PosY(1:sep_idx_train);

%Validation targets
pos_x_val = PosX(sep_idx_train+1:sep_idx_val);
pos_y_val = PosY(sep_idx_train+1:sep_idx_val);

%Test targets
pos_x_test = PosX(sep_idx_val+1:end);
pos_y_test = PosY(sep_idx_val+1:end);

%% Train, validation and train sets
train = Data(1:sep_idx_train,:);
validation = Data(sep_idx_train+1:sep_idx_val,:);
test = Data(sep_idx_val+1:end,:);

%% PCA
[std_train, mu, sigma] = zscore(train); %normalize on train data
std_validation = (validation - mu) ./ sigma;
std_test = (test - mu ) ./ sigma; %using same normalization coefficients for test data

[coeff, score, latent] = pca(std_train);
pca_train = std_train * coeff; %=score
pca_validation = std_validation * coeff;
pca_test = std_test * coeff; 

%% Choose number of PCs
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
%Conclusion: 690 features explain 90% variance.

%% Elastic nets
%From previous test, the optimal hyperparameter alpha seem to be the lowest
%one, i.e. 0.01; Reminder: alpha = ]0,1]

lambdas = logspace(-10, 0, 15);

[bx_lasso, FitInfo_lasso] = lasso(pca_train, pos_x_train, 'Lambda', lambdas, 'Alpha', 1); %alpha=1 -> Lasso
[bx_en, FitInfo_en] = lasso(pca_train, pos_x_train, 'Lambda', lambdas, 'Alpha', 0.01); 

%Train error
train_error_lasso = zeros(1, length(lambdas));
train_error_en = zeros(1, length(lambdas));
%Error estimation (MSE) using validation set
validation_error_lasso = zeros(1, length(lambdas));
validation_error_en = zeros(1, length(lambdas));

for i = 1:length(lambdas)
    coeff_lasso = [FitInfo_lasso.Intercept(i) bx_lasso(:,i)'];
    coeff_en = [FitInfo_en.Intercept(i) bx_en(:,i)'];
    
    %train
    I_train_lasso = ones(size(pos_x_train, 1), 1); 
    X_train_lasso = [I_train_lasso pca_train];
    
    I_train_en = ones(size(pos_x_train, 1), 1); 
    X_train_en = [I_train_en pca_train];
    
    %Validation
    I_val_lasso = ones(size(pos_x_val, 1), 1); 
    X_val_lasso = [I_val_lasso pca_validation]; %Use PCA set
    
    I_val_en = ones(size(pos_x_val, 1), 1); 
    X_val_en = [I_val_en pca_validation]; %Use PCA set
    
    %Train error
    mse_train_lasso = immse(pos_x_train, X_train_lasso * coeff_lasso');
    mse_train_en = immse(pos_x_train, X_train_en * coeff_en');
    
    %Validation error
    mse_val_lasso = immse(pos_x_val, X_val_lasso * coeff_lasso');
    mse_val_en = immse(pos_x_val, X_val_en * coeff_en');
    
    train_error_lasso(i) = mse_train_lasso;
    train_error_en(i) = mse_train_en;
    
    validation_error_lasso(i) = mse_val_lasso;
    validation_error_en(i) = mse_val_en;
end

%% Plot the results - Lambda and MSE
%With PCA
figure
semilogx(lambdas, train_error_lasso, '--r'); hold on;
semilogx(lambdas, validation_error_lasso, 'r');
semilogx(lambdas, train_error_en, '--b');
semilogx(lambdas, validation_error_en, 'b');
xlabel('\lambda');
ylabel('MSE');
legend('Train error - Lasso with PCA', 'Validation error - Lasso with PCA', 'Train error - Elastic nets with PCA', 'Validation error - Elastic Nets with PCA');
title('Mean-square error for different \lambda');

%% Plot the results - Lambda and MSE (this time MSE from FitInfo)
%With PCA
figure
%For the train set
semilogx(lambdas, FitInfo_lasso.MSE); hold on; 
semilogx(lambdas, FitInfo_en.MSE);
xlabel('\lambda');
ylabel('MSE');
legend('Train error - Lasso with PCA', 'Train error - Elastic nets with PCA');
title('Mean-square error for different \lambda');

%% Plot the results - Lambda and NNZ
%With PCA
figure
semilogx(lambdas, FitInfo_lasso.DF); hold on;
semilogx(lambdas, FitInfo_en.DF);
xlabel('\lambda');
ylabel('Non-zeros elements');
legend('Lasso with PCA', 'Elastic nets with PCA');
title('Number of non-zero elements for different \lambda');

