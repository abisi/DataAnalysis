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
[Bx, FitInfo_x] = lasso(train, pos_x, 'Lambda', lambda, 'CV', k);
[By, FitInfo_y] = lasso(train, pos_y, 'Lambda', lambda, 'CV', k);

%Number of non-zero beta weight
nz_weight=[]; %contain non-zero element for each lambda
for i = 1:length(lambda)
   nnz_el = nnz(Bx(:, i));
   nz_weight = [nz_weight nnz_el];
end

%as lambda increase, the number of 0 increases as well (i.e. the number of
%non-zero elements decreases)

plot(FitInfo_x.Lambda, FitInfo_x.MSE);
semilogx(lambda, FitInfo_x.MSE);
xlabel('\lambda');
ylabel('MSE');

%Selecting lambda corresponding to the best MSE
[min_mse, min_mse_idx] = min(FitInfo_x.MSE);
min_lambda = lambda(min_mse_idx)

intercept_x = FitInfo_x.Intercept(min_mse_idx)
beta_x = Bx(:, min_mse_idx);

%Regress test data


%% Elastic nets

[Bx, FitInfo_x] = lasso(train, pos_x, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);
[By, FitInfo_y] = lasso(train, pos_y, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);

