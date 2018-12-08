%% Final model building
%Here we'll use all the data set to build our final model with optimal
%parameters computed from the Guidesheets.
%Different options: With/without PCA. Lasso/Elastic nets. Optimal lambda,
%alpha. Optimal number of features.
%
%STANDARD LINEAR REGRESSION
%Number of features with PCA: 741 (corresponding to 90% variance)
%
%LASSO / ELASTIC NETS
%Number of features with PCA: 350
%lambda = 2.683e-4; 
%alpha = 0.58;

%% Data loading
clear all;
close all;
load('../data/Data.mat');

%% Standard Linear regression with PCA
%% Data set partitioning and PCA 
%Partition
train_proportion = 0.7;     
rows = size(Data,1); 
sep_idx = round(rows*train_proportion);

train = Data(1:sep_idx,:);
test = Data(sep_idx+1:end,:); 

[std_train, mu, sigma] = zscore(train); 
std_test = (test - mu ) ./ sigma; 

%PCA
[coeff, score, latent] = pca(std_train);

pca_train = std_train * coeff;
pca_test = std_test * coeff; 

%% Regression - linear
nPCs = 741;  %for 90% total variance
%Train
pos_x_train = PosX(1:sep_idx); 
pos_y_train = PosY(1:sep_idx); 
FM_train = pca_train(:,1:nPCs); 
I_train = ones(size(FM_train,1),1); 
X_train = [I_train FM_train];

%Regression and mse
bx = regress(pos_x_train, X_train); 
by = regress(pos_y_train, X_train);
x_hat = X_train * bx; 
y_hat = X_train * by;
mse_pos_x = immse(pos_x_train, x_hat); 
mse_pos_y = immse(pos_y_train, y_hat);

%Test
pos_x_test = PosX(sep_idx+1:end); 
pos_y_test = PosY(sep_idx+1:end);
FM_test = pca_test(:,1:nPCs);
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

x_hat_test = X_test * bx; 
y_hat_test = X_test * by;
mse_posx_test = immse(pos_x_test, x_hat_test);
mse_posy_test = immse(pos_y_test, y_hat_test);

%% Plot - Linear regression
regressed_x = [x_hat; x_hat_test];
regressed_y = [y_hat; y_hat_test];

%X motion
subplot(2,1,1)
plot(PosX, 'Linewidth', 1); hold on;
plot(regressed_x, 'LineWidth', 1); hold off;
xlabel('Time [ms]');
ylabel('X');
axis([8500 9500 -0.05 0.18]); %2000-2500 vs -0.05 0.18
x = [sep_idx sep_idx];
y = [-1 1];
line(x, y, 'Color', 'black', 'LineStyle', '--')
legend('Observed', 'Predicted', 'Train/Test separation');
title('Linear regression with PCA - Position X');

%Y motion
subplot(2,1,2)
plot(PosY, 'LineWidth', 1); hold on;
plot(regressed_y, 'LineWidth', 1); hold off;
xlabel('Time [ms]');
ylabel('Y');
axis([8500 9500 0.1 0.33]); %2000-2500 vs -0.05 0.18
line(x, y, 'Color', 'black', 'LineStyle', '--')
legend('Observed', 'Predicted', 'Train/Test separation');
title('Linear regression with PCA - Position Y');

%% Stock variables for final graph
regressed_x_linear = regressed_x;
regressed_y_linear = regressed_y;

%% Display MSE
disp(['Linear regression: MSE train (Position X) = ', num2str(mse_pos_x)]);
disp(['Linear regression: MSE train (Position Y) = ', num2str(mse_pos_y)]);
disp(['Linear regression: MSE test (Position X) = ', num2str(mse_posx_test)]);
disp(['Linear regression: MSE test (Position Y) = ', num2str(mse_posy_test)]);

%% Lasso / Elastic Nets
%% Clean variables
clear all;
close all;
load('../data/Data.mat');

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
nPCs = 350;

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



%% Lasso / Elastic Nets - Plot results
regressed_x = [x_hat_train; x_hat_test];
regressed_y = [y_hat_train; y_hat_test];

figure
subplot(2,1,1)
plot(PosX, 'LineWidth', 1.5); hold on;
plot(regressed_x, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position X');
a = [sep_idx sep_idx];
b = [-1 1];
line(a, b, 'Color', 'black', 'LineStyle', '--')
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8500 9500 -0.05 0.18]);
title(['Elastic Nets with PCA (\alpha = ', num2str(alpha), ')', ' - Position X']);

subplot(2,1,2);
plot(PosY, 'LineWidth', 1.5); hold on;
plot(regressed_y, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position Y');
line(a, b, 'Color', 'black', 'LineStyle', '--')
legend('Observed', 'Predicted', 'Train/Test separation');
axis([8500 9500 0.15 0.30]);
title(['Elastic Nets with PCA (\alpha = ', num2str(alpha), ')', ' - Position Y']);

%% Stock variables for final graph
regressed_x_en = regressed_x;
regressed_y_en = regressed_y;

%% Display results
disp(['EN/Lasso: MSE train (Position X) (alpha = ', num2str(alpha), ') = ', num2str(mse_x_train)]);
disp(['EN/Lasso: MSE train (Position Y) (alpha = ', num2str(alpha), ') = ', num2str(mse_y_train)]);
disp(['EN/Lasso: MSE test (Position X) (alpha = ', num2str(alpha), ') = ', num2str(mse_x_test)]);
disp(['EN/Lasso: MSE test (Position Y) (alpha = ', num2str(alpha), ') = ', num2str(mse_y_test)]);

%% Final graph
% a = [sep_idx sep_idx];
% b = [-1 1];
% 
% figure
% subplot(2,2,1)
% plot(PosX, 'LineWidth', 1.5); hold on;
% plot(regressed_x_linear, 'LineWidth', 1.5);
% xlabel('Time [ms]');
% ylabel('Position X');
% line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
% legend('Observed', 'Predicted', 'Train/Test separation');
% axis([8500 9500 0 0.18]);
% title('Linear regression with PCA - Position X');
% 
% subplot(2,2,2)
% plot(PosX, 'LineWidth', 1.5); hold on;
% plot(regressed_y_linear, 'LineWidth', 1.5);
% xlabel('Time [ms]');
% ylabel('Position Y');
% line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
% legend('Observed', 'Predicted', 'Train/Test separation');
% axis([8500 9500 0 0.18]);
% title('Linear regression with PCA - Position Y');
% 
% subplot(2,2,3);
% plot(PosX, 'LineWidth', 1.5); hold on;
% plot(regressed_x_en, 'LineWidth', 1.5);
% xlabel('Time [ms]');
% ylabel('Position X');
% line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
% legend('Observed', 'Predicted', 'Train/Test separation');
% axis([8500 9500 0.15 0.28]);
% title(['Elastic Nets with PCA (\alpha = ', num2str(alpha), ')', ' - Position X']);
% 
% subplot(2,2,4);
% plot(PosY, 'LineWidth', 1.5); hold on;
% plot(regressed_y_en, 'LineWidth', 1.5);
% xlabel('Time [ms]');
% ylabel('Position Y');
% line(a, b, 'Color', 'black', 'LineStyle', '--'); hold off;
% legend('Observed', 'Predicted', 'Train/Test separation');
% axis([8500 9500 0.15 0.28]);
% title(['Elastic Nets with PCA (\alpha = ', num2str(alpha), ')', ' - Position Y']);

