%% Final model building
%Here we'll use all the data set to build our final model with optimal
%parameters computed from the Guidesheets.
%Different options: With/without PCA. Lasso/Elastic nets. Optimal lambda,
%alpha. Optimal number of features.
%
%STANDARD LINEAR REGRESSION
%Number of features with PCA: 690.
%
%LASSO / ELASTIC NETS
%Number of features with PCA: ?
%lambda = 10e-4; (use only this if we use lasso)
%alpha = ?

%Questions:
%Is the number of PCs the same for the position X and Y? Should not.

%% Data loading
clear all;
close all;
load('../data/Data.mat');

%% Standard Linear regression



%% Lasso / Elastic Nets
lambda = 10e-4;
nPCs_x = 300;
nPCs_y = 290;
%PCA
[coeff, score, latent] = pca(Data);
%Lasso
[bx, FitInfox] = lasso(score(:, 1:nPCs_x), PosX, 'Lambda', lambda);
[by, FitInfoy] = lasso(score(:, 1:nPCs_y), PosY, 'Lambda', lambda);
%Prediction X
FM_x = [score(:,1:nPCs_x)];
I_x = ones(size(FM_x,1),1);
X_x = [I_x FM_x];

Bx = [FitInfox.Intercept bx'];

x_hat = X_x * Bx';

%Prediction Y
FM_y = [score(:,1:nPCs_y)];
I_y = ones(size(FM_y,1),1);
X_y = [I_y FM_y];

By = [FitInfoy.Intercept by'];

y_hat = X_y * By';

%% Lasso - Plot results
figure
subplot(2,1,1)
plot(PosX, 'LineWidth', 1.5); hold on;
plot(x_hat, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position X');
legend('Observed', 'Predicted');
axis([2000 3000 -0.05 0.15]);
title('Observed and predicted X position - Lasso with PCA');

subplot(2,1,2);
plot(PosY, 'LineWidth', 1.5); hold on;
plot(y_hat, 'LineWidth', 1.5);
xlabel('Time [ms]');
ylabel('Position Y');
legend('Observed', 'Predicted');
axis([2000 3000 0.15 0.28]);
title('Observed and predicted Y position - Lasso with PCA');
