%% Alpha optimisation
%% Data loading
%The aim here is to study under which condition
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

%% alpha, lambda, percentage train
%Take posx for all the following 
lambdas = logspace(-10, 0, 15);
%alphas = 0.01:0.1:1; %because 'Alpha' must be in the interval (0.1]
alphas = linspace(0.01, 1, length(lambdas));

optimal_lambda_storage = zeros(1, length(alphas));
min_mse_storage = zeros(1, length(alphas));

optimal_alpha = 0;

%Number of non-zero elements
DF_storage = zeros(1, length(alphas));
DF_x = 0;

%Grid search
S = zeros(length(lambdas), length(alphas));

for i = 1:length(alphas)
   [bx, FitInfo_x] = lasso(train, pos_x_train, 'Lambda', lambdas, 'Alpha', alphas(i));  
  
   validation_error_storage = zeros(1, length(lambdas)); 
   
   for j = 1:length(lambdas)
       coeff_x = [FitInfo_x.Intercept(j) bx(:,j)'];
       
       %Validation error
       I_val = ones(size(pos_x_val, 1), 1); 
       X_val = [I_val validation];
       
       mse_x = immse(pos_x_val, X_val * coeff_x');
       
       validation_error_storage(j) = mse_x;
       
       %Grid search
       S(j, i) = mse_x;
   end
   
   [min_mse_x, min_mse_idx] = min(validation_error_storage);
   opt_lambda_x = lambdas(min_mse_idx);
   
   optimal_lambda_storage(i) = opt_lambda_x;
   min_mse_storage(i) = min_mse_x;
   
   DF_x = FitInfo_x.DF(min_mse_idx);
   DF_storage(i) = DF_x;
end

%Alpha optimal - Encore une fois le meilleur alpha correspon � l'erreur min
[optimal_mse, optimal_mse_idx] = min(min_mse_storage);
optimal_alpha = alphas(optimal_mse_idx);

%Lambda optimal
optimal_lambda = optimal_lambda_storage(optimal_mse_idx);

%DF (number of non-zero elements) corresponfing to best alpha
DF_final = DF_storage(optimal_mse_idx);

%% Display the results
disp(['Best MSE (x) = ', num2str(optimal_mse)]);
disp(['Best alpha (x) = ', num2str(optimal_alpha)]);
disp(['Best lambda (x) = ', num2str(optimal_lambda)]);
disp(['Number of non-zero elements (x) = ', num2str(DF_final)]);

%% Plot results
figure()
plot(alphas, min_mse_storage);
xlabel('\alpha');
ylabel('MSE');

figure()
plot(alphas, DF_storage);
xlabel('\alpha');
ylabel('Non-zero elements');

figure()
plot(alphas, optimal_lambda_storage);
xlabel('\alpha');
ylabel('\lambda');

%% Grid search
figure
surf(lambdas,alphas,S);
set(gca,'XScale','log');
view(2)
colorbar
xlabel('\lambda');
ylabel('\alpha');

%%
% figure
% %colormap parula;
% imagesc([lambdas(1) lambdas(end)],[alphas(1) alphas(end)], S); hold on; 
% %surf(lambdas, alphas);
% set(gca, 'Ydir', 'normal');    
% %set(gca, 'xscale', 'log');
% %colorbar('Ticks', [0 2], 'TickLabels', {'Phase derive','Synchronised'});
% xlabel('\lambda');
% ylabel('\alpha');
% colorbar
