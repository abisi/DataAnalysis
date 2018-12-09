%% Linear regression
%% Clean variables
clear all;
close all;
load('../data/Data.mat');

%%
proportion = 0.7;     
rows = size(Data,1); 
sep_idx = round(rows*proportion);
train = Data(1:sep_idx,:);
test = Data(sep_idx:end,:); %here we keep order because we wanna predict future values based on past values

%Stdize
[std_train, mu, sigma] = zscore(train); 
std_test = (test - mu ) ./ sigma; %using same normalization coefficients

%Train
target_posx = PosX(1:sep_idx); % x position reg. targets
target_posy = PosY(1:sep_idx); % y position reg. targets

%Test
target_posx_te = PosX(sep_idx:end); 
target_posy_te = PosY(sep_idx:end);

%%
FM_train = std_train; % training feature matrix
I_train = ones(size(FM_train,1),1); % X should include a column of ones so that the model contains a constant term
X_train = [I_train FM_train];

%Regression and mse
bx = regress(target_posx, X_train); % coefficients b 
by = regress(target_posy, X_train);
x_hat = X_train * bx; %regression vectors
y_hat = X_train * by;
mse_posx = immse(target_posx, x_hat); %mse
mse_posy = immse(target_posy, y_hat);

%Test
FM_test = std_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

x_hat_te = X_test * bx; %using SAME coefficients
y_hat_te = X_test * by;
mse_posx_test = immse(target_posx_te, x_hat_te);
mse_posy_test = immse(target_posy_te, y_hat_te);

%% Errors
mse_posx = immse(target_posx, x_hat); 
mse_posy = immse(target_posy, y_hat);
mse_posx_te = immse(target_posx_te, x_hat_te);
mse_posy_te = immse(target_posy_te, y_hat_te);
errors_linear_n = [mse_posx, mse_posy, mse_posx_te, mse_posy_te];
errors_linear_rmse = sqrt(errors_linear);
figure
bar(errors_linear_n);

%% Stock variables for final graph
regressed_x_linear = [x_hat; x_hat_te];
regressed_y_linear = [y_hat; y_hat_te];

%% Polynomial regression
%%
FM_train = std_train; % training feature matrix
I_train = ones(size(FM_train,1),1); 
X_train = [I_train FM_train FM_train.^2];
FM_test = std_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test FM_test.^2];

bx = regress(target_posx, X_train);  
by = regress(target_posy, X_train);
x_hat = X_train * bx; 
y_hat = X_train * by;
mse_posx = immse(target_posx, x_hat); 
mse_posy = immse(target_posy, y_hat);
    
x_hat_te = X_test * bx; 
y_hat_te = X_test * by;
mse_posx_te = immse(target_posx_te, x_hat_te); 
mse_posy_te = immse(target_posy_te, y_hat_te);

%% Stock variables for final graph
regressed_x_linear_2 = [x_hat; x_hat_te];
regressed_y_linear_2 = [y_hat; y_hat_te];

%% Lasso / Elastic Nets
%% Data partiotionning
%Partition
train_proportion = 0.7;     
rows = size(Data,1); 
sep_idx = round(rows*train_proportion);

train = Data(1:sep_idx,:);
test = Data(sep_idx+1:end,:); 

pos_x_train = PosX(1:sep_idx); 
pos_y_train = PosY(1:sep_idx); 

pos_x_test = PosX(sep_idx+1:end); 
pos_y_test = PosY(sep_idx+1:end);

%Standardization
[std_train, mu, sigma] = zscore(train); 
std_test = (test - mu ) ./ sigma; 

%PCA
[coeff, score, latent] = pca(std_train);

pca_train = std_train * coeff;
pca_test = std_test * coeff;

%% Regularized regression
lambda = 2.68e-4;
alpha = 0.57;
nPCs = 390;

[bx, FitInfox] = lasso(pca_train(:, 1:nPCs), pos_x_train, 'Lambda', lambda, 'Alpha', alpha);
[by, FitInfoy] = lasso(pca_train(:, 1:nPCs), pos_y_train, 'Lambda', lambda, 'Alpha', alpha);

coeff_x = [FitInfox.Intercept bx'];
coeff_y = [FitInfoy.Intercept by'];

%Train
FM_train = pca_train(:, 1:nPCs);
I_train = ones(size(FM_train, 1), 1); 
X_train = [I_train FM_train]; 

%Test
FM_test = pca_test(:, 1:nPCs); 
I_test = ones(size(FM_test, 1), 1); 
X_test = [I_test FM_test];

x_hat_train = X_train * coeff_x';
y_hat_train = X_train * coeff_y';

x_hat_test = X_test * coeff_x';
y_hat_test = X_test * coeff_y';

%Train error
mse_x_train = immse(pos_x_train, x_hat_train);
mse_y_train = immse(pos_y_train, y_hat_train);
    
%Test error
mse_x_test = immse(pos_x_test, x_hat_test);
mse_y_test = immse(pos_y_test, y_hat_test);

%% Stock variables for final graph
regressed_x_en = [x_hat_train; x_hat_test];
regressed_y_en = [y_hat_train; y_hat_test];

%% Display results
disp(['EN/Lasso: MSE train (Position X) (alpha = ', num2str(alpha), ')', '(nPCs = ', num2str(nPCs), ') = ', num2str(mse_x_train)]);
disp(['EN/Lasso: MSE train (Position Y) (alpha = ', num2str(alpha), ')', '(nPCs = ', num2str(nPCs), ') = ', num2str(mse_y_train)]);
disp(['EN/Lasso: MSE test (Position X) (alpha = ', num2str(alpha), ')', '(nPCs = ', num2str(nPCs), ') = ', num2str(mse_x_test)]);
disp(['EN/Lasso: MSE test (Position Y) (alpha = ', num2str(alpha), ')', '(nPCs = ', num2str(nPCs), ') = ', num2str(mse_y_test)]);

%% Final graph
a = [sep_idx sep_idx];
b = [-1 1];

figure
subplot(3,2,1)
plot(PosX, 'LineWidth', 1.5); hold on; %'Color', c1
plot(regressed_x_linear, 'LineWidth', 1.5); %'Color', c2
xlabel('Time [ms]');
ylabel('Position X');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0 0.18]);
title('\fontsize{14} Linear regression w/o PCA - Position X'); %\fontsize{14}

subplot(3,2,2)
plot(PosY, 'LineWidth', 1.5); hold on;
plot(regressed_y_linear, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position Y');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0.15 0.30]);
title('\fontsize{14} Linear regression w/o PCA - Position Y');

subplot(3,2,3);
plot(PosX, 'LineWidth', 1.5); hold on;
plot(regressed_x_linear_2, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position X');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0 0.18]);
title(['\fontsize{14} Polynomial regression (\it{i=2}) w/o PCA (\alpha = ', num2str(alpha), ')', ' - Position X']);

subplot(3,2,4);
plot(PosY, 'LineWidth', 1.5); hold on;
plot(regressed_y_linear_2, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position Y');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0.15 0.30]);
title(['\fontsize{14} Polynomial regression (\it{i=2}) w/o PCA (\alpha = ', num2str(alpha), ')', ' - Position Y']);

subplot(3,2,5);
plot(PosX, 'LineWidth', 1.5); hold on;
plot(regressed_x_en, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position X');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0 0.18]);
title(['\fontsize{14} Elastic Net with PCA (\alpha = ', num2str(alpha), ')', ' - Position X']);

subplot(3,2,6);
plot(PosY, 'LineWidth', 1.5); hold on;
plot(regressed_y_en, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position Y');
line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8700 9100 0.15 0.30]);
title(['\fontsize{14} Elastic Net with PCA (\alpha = ', num2str(alpha), ')', ' - Position Y']);