%% Guidesheet 8

% --- Questions for TAs ---
%REGRESSION: 
% -Do we train with entire data set's features?
%LASSO:
% -How do we do CV if cannot mix samples because time-ordered ? Do we only do
% CV on same 5% of data set (train), then validate everytime on remaining
% following 65% ?
% -Does it really add an intercept to training data X ? Because it
% gives back B which are a size different than X = [I FM]...
% -Is it normal that validation and testing errors are roughly the same ?
%Final model:
%How should we separate our data? (as usual: 60-30-10)?
%Conclusion, relevant point to mention in the report:
%PCA needed?

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

nFeatures = size(train,2); %TODO: define appropriate feature number ? Right now we do with everything

%Train: for PosX and PosY
FM_train = [train(:, 1:nFeatures)];
I_train = ones(size(pos_x, 1), 1);
X_train = [I_train FM_train];

bx = regress(pos_x, X_train);
by = regress(pos_y, X_train);

x_hat = X_train * bx; %regression vectors
y_hat = X_train * by;

mse_posx_train = immse(pos_x, x_hat); 
mse_posy_train = immse(pos_y, y_hat); 

mse_errors_train = [mse_posx_train mse_posy_train];
disp(mse_errors_train)

%Test:for PosX and PosY
%Here FM = validation + test -> 95% is test!
FM_test = Data(sep_idx_train+1:end,:);
I_test = ones(floor((1-train_proportion) * rows), 1);
X_test = [I_test FM_test];

x_hat_test = X_test * bx; %use same parameters (from training)
y_hat_test = X_test * by;

mse_posx_test = immse(pos_x_test, x_hat_test); 
mse_posy_test = immse(pos_y_test, y_hat_test); 

mse_errors_test = [mse_posx_test mse_posy_test];
disp(mse_errors_test)

%Errors of test are much bigger than train - overfitting
% x: 0.0073   y: 0.0061
% Are they good ?  


%% Plot output coordinates compared labels

regressed_x = [x_hat; x_hat_test];
regressed_y = [y_hat; y_hat_test];

figure
subplot(1,2,1)
plot(x_hat, y_hat, 'LineWidth', 2); hold on
plot(pos_x, pos_y); hold off
xlabel('Position X')
ylabel('Postion Y')
title('Predicted and real movements of monkey''s wrist - train test')
legend('Predicted trajectory','Real trajectory')

subplot(1,2,2)
plot(x_hat_test, y_hat_test, 'LineWidth', 2); hold on
plot(pos_x_test, pos_y_test); hold off
xlabel('Position X')
ylabel('Postion Y')
title('Predicted and real movements of monkey''s wrist - test test')
legend('Predicted trajectory','Real trajectory')

% Strong overfitting of the test set ! 

%Plot the position as a function of time > to compare with Guidesheet7
%i find personnally the output more clear to interpret
%TRAIN
figure
subplot(2,1,1)
plot(x_hat, 'LineWidth', 1); hold on;
plot(pos_x, '--r', 'LineWidth', 1); hold off;
xlabel('Time [ms]');
ylabel('X');
title('Arm and predicted trajectory along x');
legend('Predicted trajectory','Real trajectory');

subplot(2,1,2)
plot(y_hat, 'LineWidth', 1); hold on;
plot(pos_y, '--r', 'LineWidth', 1); hold off;
xlabel('Time [ms]');
ylabel('Y');
title('Arm and predicted trajectory along y');
legend('Predicted trajectory','Real trajectory');

%TEST
figure
subplot(2,1,1)
plot(x_hat_test, 'LineWidth', 1); hold on
plot(pos_x_test, '--r', 'LineWidth', 1); hold off
xlabel('Time [ms]')
ylabel('X')
axis([3500 4000 -0.4 0.6]);
title('Predicted and real movements of monkey''s wrist - test test')
legend('Predicted trajectory','Real trajectory')

subplot(2,1,2)
plot(y_hat_test, 'LineWidth', 1); hold on
plot(pos_y_test, '--r', 'LineWidth', 1); hold off
xlabel('Time [ms]')
ylabel('Y')
axis([3500 4000 -0.15 0.6]);
title('Predicted and real movements of monkey''s wrist - test test')
legend('Predicted trajectory','Real trajectory')

%% LASSO
%For the purpose of the exercice we'll split the data in 3 sets:
%5% training, 65% validation, 30% test

%Train targets
pos_x_train = PosX(1:sep_idx_train);
pos_y_train = PosY(1:sep_idx_train);

%Validation targets
pos_x_val = PosX(sep_idx_train+1:sep_idx_val);
pos_y_val = PosY(sep_idx_train+1:sep_idx_val);

%Test targets
pos_x_test = PosX(sep_idx_val+1:end);
pos_y_test = PosY(sep_idx_val+1:end);

%Parameters
lambda = logspace(-10, 0, 30);
k = 10;

%Since lasso arleady does CV we feed it with both train and validation
[Bx, FitInfo_x] = lasso(train, pos_x_train, 'Lambda', lambda, 'CV', k);
[By, FitInfo_y] = lasso(train, pos_y_train, 'Lambda', lambda, 'CV', k);


%% Number of non-zero beta weights
nz_weight=[]; %contains non-zero element for each lambda
for i = 1:length(lambda)
   nnz_el = nnz(Bx(:, i));
   nz_weight = [nz_weight nnz_el];
end

figure
plot(lambda, nz_weight);  
semilogx(lambda, nz_weight, 'LineWidth', 1.5);
xlabel('\lambda')
ylabel('Number of non-zero weights')
title('Sparsity of weights \beta')

% Number of ZERO weights greatly decreases as lambda increases.

%% Selecting lambda corresponding to the best MSE
%X
[min_mse_x, min_mse_idx_x] = min(FitInfo_x.MSE);
min_lambda_x = lambda(min_mse_idx_x)
intercept_x = FitInfo_x.Intercept(min_mse_idx_x)
beta_x = Bx(:, min_mse_idx_x);

%Y - OR, should we keep the same lambda min ? 
[min_mse_y, min_mse_idx_y] = min(FitInfo_y.MSE);
min_lambda_y = lambda(min_mse_idx_y)
intercept_y = FitInfo_y.Intercept(min_mse_idx_y)
beta_y = By(:, min_mse_idx_y);

% Now, regress test data

FM_val = Data(sep_idx_train+1:sep_idx_val,:);
I_val = ones(size(FM_val,1),1); %not used because lasso apparently does not add one... despite what the guidesheet says?
X_val = [I_val FM_val];

x_hat_lasso = X_val * beta_x; %regression vectors: weight for each feature for lambda yielding lowest MSE
y_hat_lasso = X_val * beta_y;

%Compute validation errors
mse_posx_val = immse(pos_x_val, x_hat_lasso); 
mse_posy_val = immse(pos_y_val, y_hat_lasso); 

val_errors = [mse_posx_val mse_posy_val]

% Validation errors
% X: 0.0120
% Y: 0.0248

%% Now on test set - LASSO

FM_test = Data(sep_idx_val+1:end,:);
X_test = [FM_test]; %useless I know but jsut to keep same structure

%Predict
x_hat_lasso_test = X_test * beta_x; %regression vectors: weight for each feature for lambda yielding lowest MSE
y_hat_lasso_test = X_test * beta_y;

%Compute test errors
mse_posx_test = immse(pos_x_test, x_hat_lasso_test); 
mse_posy_test = immse(pos_y_test, y_hat_lasso_test); 

test_errors = [mse_posx_test mse_posy_test]

%Test errors
% X: 0.0112
% Y: 0.0250 
%Validation and test errors are roughly the same... normal ? 

%% Plots LASSO

%As lambda increases, the number of 0 increases as well (i.e. the number of
%non-zero elements decreases)

figure
subplot(1,2,1)
h(1) = plot(FitInfo_x.Lambda, FitInfo_x.MSE); 
h(2) = semilogx(lambda, FitInfo_x.MSE, 'LineWidth', 1.5); hold on
h(3) = semilogx(lambda, FitInfo_x.MSE, '*', 'MarkerIndices', [FitInfo_x.IndexMinMSE], ...
    'MarkerFaceColor','red', 'MarkerSize', 10);
xlabel('\lambda')
ylabel('MSE')
title('Position X')
legend(h([3]),'Minimum MSE')

subplot(1,2,2)
h(1) = plot(FitInfo_y.Lambda, FitInfo_y.MSE); 
h(2) = semilogx(lambda, FitInfo_y.MSE, 'LineWidth', 1.5); hold on
h(3) = semilogx(lambda, FitInfo_y.MSE, '*', 'MarkerIndices', [FitInfo_y.IndexMinMSE], ...
    'MarkerFaceColor','red', 'MarkerSize', 10); 
xlabel('\lambda')
ylabel('MSE')
title('Position Y')
legend(h([3]),'Minimum MSE')


%% Elastic nets

[Bx_en, FitInfo_x_en] = lasso(train, pos_x, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);
[By_en, FitInfo_y_en] = lasso(train, pos_y, 'Lambda', lambda, 'CV', k, 'Alpha', 0.5);

%% Number of non-zero beta weights
nz_weight_en=[]; %contains non-zero element for each lambda
for i = 1:length(lambda)
   nnz_el = nnz(Bx_en(:, i));
   nz_weight_en = [nz_weight_en nnz_el];
end

figure
plot(lambda, nz_weight_en);  
semilogx(lambda, nz_weight_en, 'LineWidth', 1.5);
xlabel('\lambda')
ylabel('Number of non-zero weights')
title('Sparsity of weights \beta')

% Number of ZERO weights greatly decreases as lambda increases.

%% Selecting lambda corresponding to the best MSE
%X
[min_mse_x_en, min_mse_idx_x_en] = min(FitInfo_x_en.MSE);
min_lambda_x_en = lambda(min_mse_idx_x_en)
intercept_x_en = FitInfo_x_en.Intercept(min_mse_idx_x_en)
beta_x_en = Bx(:, min_mse_idx_x_en);

%Y - OR, should we keep the same lambda min ? 
[min_mse_y_en, min_mse_idx_y_en] = min(FitInfo_y_en.MSE);
min_lambda_y_en = lambda(min_mse_idx_y_en)
intercept_y_en = FitInfo_y_en.Intercept(min_mse_idx_y_en)
beta_y_en = By(:, min_mse_idx_y_en);

% Now, regress test data

FM_val_en = Data(sep_idx_train+1:sep_idx_val,:);
I_val_en = ones(size(FM_val,1),1); %not used because lasso apparently does not add one... despite what the guidesheet says?
X_val_en = [I_val_en FM_val_en];

x_hat_en = X_val_en * beta_x_en; %regression vectors: weight for each feature for lambda yielding lowest MSE
y_hat_en = X_val_en * beta_y_en;

%Compute validation errors
mse_posx_val_en = immse(pos_x_val, x_hat_en); 
mse_posy_val_en = immse(pos_y_val, y_hat_en); 

val_errors = [mse_posx_val_en mse_posy_val_en]

%% Elestic net on test set




%% Comparisaon with Lasso



%% Hyperparameters optimization
%Test for several values of alpha
%lambda still logspace(...)

%Questions for TAs
%Hyperparams optimisation > cross-validation or single training-test?
%Performance estimation?
k = 10;

%Adding the L2 norm constraint
alpha = 0.01:0.01:1; %0.01 because 'Alpha' must be in the interval (0.1]
%Note: very long to compute, try first with 0.1 instead of 0.01

optimal_lambda_x_storage = zeros(1, length(alpha)); %pour chaque alpha on aura un lambda optimal
optimal_lambda_y_storage = zeros(1, length(alpha));

min_MSE_x_storage = zeros(1, length(alpha));
min_MSE_y_storage = zeros(1, length(alpha));

optimal_alpha_x = 0;
optimal_alpha_y = 0;

%Number of non-zero elements
DF_x_storage = zeros(1, length(alpha));
DF_y_storage = zeros(1, length(alpha));
DF_x = 0;
DF_y = 0;

for i = 1:length(alpha)
   [bx, FitInfo_x] = lasso(train, pos_x, 'Lambda', lambda, 'CV', k, 'Alpha', alpha(i)); 
   [by, FitInfo_y] = lasso(train, pos_y, 'Lambda', lambda, 'CV', k, 'Alpha', alpha(i));
   
   %Find min MSE and the corresponding lambda
   [min_MSE_x, min_MSE_idx] = min(FitInfo_x.MSE);
   opt_lambda_x = FitInfo_x.Lambda(min_MSE_idx);
   
   optimal_lambda_x_storage = [optimal_lambda_x_storage opt_lambda_x];
   min_MSE_x_storage = [min_MSE_x_storage min_MSE_x];
   
   DF_x = FitInfo_x.DF(min_MSE_idx);
   DF_x_storage = [DF_x_storage DF_x];
   
   [min_MSE_y, min_MSE_idy] = min(FitInfo_y.MSE);
   opt_lambda_y = FitInfo_y.Lambda(min_MSE_idy);
   
   optimal_lambda_y_storage = [optimal_lambda_y_storage opt_lambda_y];
   min_MSE_y_storage = [min_MSE_y_storage min_MSE_y];
   
   DF_y = FitInfo_y.DF(min_MSE_idy);
   DF_y_storage = [DF_y_storage DF_y];
   
end

%Alpha optimal - Encore une fois le meilleur alpha correspon à l'erreur min
[optimal_MSE_x, optimal_MSE_x_idx] = min(min_MSE_x_storage);
optimal_alpha_x = alpha(optimal_MSE_x_idx);

[optimal_MSE_y, optimal_MSE_y_idx] = min(min_MSE_y_storage);
optimal_alpha_y = alpha(optimal_MSE_y_idx);

%Lambda optimal
optimal_lambda_x = optimal_lambda_x_storage(optimal_MSE_x_idx);
optimal_lambda_y = optimal_lambda_y_storage(optimal_MSE_y_idy);

%DF (number of non-zero elements) corresponfing to best alpha
DF_x_final = DF_x_storage(optimal_MSE_x_idx);
DF_y_final = DF_y_storage(optimal_MSE_y_idx);

%Display the results
disp(['Best MSE (x) = ', optimal_MSE_x]);
disp(['Best \alpha (x) = ', optimal_alpha_x]);
disp(['Best \lambda (x) = ', optimal_lambda_x]);
disp(['Number of non-zero elements (x) = ', DF_x_final]);

disp(['Best MSE (y) = ', optimal_MSE_y]);
disp(['Best \alpha (y) = ', optimal_alpha_y]);
disp(['Best \lambda (y) = ', optimal_lambda_y]);
disp(['Number of non-zero elements (y) = ', DF_y_final]);
