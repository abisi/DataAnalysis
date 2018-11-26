%% Guidesheet 8

clear all;
close all;
load('../data/Data.mat');

%% Regression

%Data partitionning
proportion = 0.05; %use 5% of the dataset for training     
rows = size(Data,1);
sep_idx = round(rows*proportion);
train = Data(1:sep_idx,:);
test = Data(sep_idx:end,:);

%Train outputs
pos_x = PosX(1:sep_idx);
pos_y = PosY(1:sep_idx);

%Test outputs
pos_x_test = PosX(sep_idx:end);
pos_y_test = PosY(sep_idx:end);

%Regression
nFeatures = 100; %TODO: define appropriate value
%Train: for PosX and PosY
FM_train = train(:, 1:nFeatures);
I_train = ones(size(pos_x, 1), 1);
X_train = [I_train FM_train];

bx = regress(pos_x, X_train(:,1:nFeatures));
by = regress(pos_y, X_train(:,1:nFeatures));

x_hat = X_train(:,1:nFeatures) * bx; %regression vectors
y_hat = X_train(:,1:nFeatures) * by;

mse_posx = immse(pos_x, x_hat); 
mse_posy = immse(pos_y, y_hat); 

%Test:for PosX and PosY
FM_test = test(:, 1:nFeatures);
I_test = ones(size(pos_x_test, 1), 1);
X_test = [I_test FM_test];

x_hat_test = X_test(:, 1:nFeatures) * bx;
y_hat_test = X_test(:, 1:nFeatures) * by;

mse_posx_test = immse(pos_x_test, x_hat_test); 
mse_posy_test = immse(pos_y_test, y_hat_test); 

%plot...


%% LASSO
lambda = logspace(-10, 0, 15);
k = 10;
[Bx, FitInfo_x, 'Lambda', lambda, 'CV', k] = lasso(train, pos_x);
[By, FitInfo_y, 'Lambda', lambda, 'CV', k] = lasso(train, pos_y);

%% Elastic nets

[Bx, FitInfo_x, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5] = lasso(train, pos_x);
[By, FitInfo_y, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5] = lasso(train, pos_y);

