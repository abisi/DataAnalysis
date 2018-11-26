%% Guidesheet 8

clear all;
close all;
load('../data/Data.mat');

%% Data partitionning
%For the report: change the values!
train_proportion = 0.05; %use 5% of the dataset for training
validation_proportion = 0.65;
rows = size(Data,1);
sep_idx_train = ceil(rows*train_proportion);
sep_idx_val = ceil(rows*(train_proportion + validation_proportion));

%% Train vs Test sets
train = Data(1:sep_idx_train,:);
validation = Data(sep_idx_train+1:sep_idx_val,:);
test = Data(sep_idx_val+1:end,:);

%Train outputs
pos_x = PosX(1:sep_idx_train);
pos_y = PosY(1:sep_idx_train);

%Test outputs
pos_x_test = PosX(sep_idx_train+1:end);
pos_y_test = PosY(sep_idx_train+1:end);

%% Regression
%For the purpose of the exercice we'll split thee data in 3 sets:
%5% training, 65% validation, 30% test
nFeatures = size(train,2); %TODO: define appropriate feature number
%Train: for PosX and PosY
FM_train = train(:, 1:nFeatures);
I_train = ones(size(pos_x, 1), 1);
X_train = [I_train FM_train];

bx = regress(pos_x, X_train);
by = regress(pos_y, X_train);

x_hat = X_train * bx; %regression vectors
y_hat = X_train * by;

mse_posx_train = immse(pos_x, x_hat); 
mse_posy_train = immse(pos_y, y_hat); 

%Test:for PosX and PosY
%FM = validation + test
FM = Data(sep_idx_train+1:end,:);
I = ones(floor((1-train_proportion) * rows), 1);
disp(size(FM))
disp(size(I))
X_test = [I FM];

x_hat_test = X_test * bx;
y_hat_test = X_test * by;

mse_posx_test = immse(pos_x_test, x_hat_test); 
mse_posy_test = immse(pos_y_test, y_hat_test); 

%plot
figure()
scatter(pos_x, pos_y, 'k', 'LineWidth', 4); hold on;
scatter(x_hat, y_hat, 'g'); hold off;

figure()
scatter(pos_x_test(1:50:end), pos_y_test(1:50:end), 'k', 'LineWidth', 4); hold on;
scatter(x_hat_test(1:50:end), y_hat_test(1:50:end), 'r');

%probable overfit


%% LASSO

%Data partitionning
%Train outputs
pos_x = PosX(1:sep_idx_train);
pos_y = PosY(1:sep_idx_train);

%Validation outputs
pox_x_val = PosX(sep_idx_train+1:sep_idx_val);
pos_y_val = PosY(sep_idx_train+1:sep_idx_val);

%Test outputs
pos_x_test = PosX(sep_idx_val+1:end);
pos_y_test = PosY(sep_idx_val+1:end);

lambda = logspace(-10, 0, 15);
k = 10;
[Bx, FitInfo_x] = lasso(train, pos_x_train, 'Lambda', lambda, 'CV', k);
[By, FitInfo_y] = lasso(train, pos_y_train, 'Lambda', lambda, 'CV', k);

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

[Bx_en, FitInfo_x_en] = lasso(train, pos_x, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);
[By_en, FitInfo_y_en] = lasso(train, pos_y, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);