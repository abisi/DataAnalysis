%% Comparing different regression
% README: In this file I first to simple linear regression with PCA, then
% regression of different degrees with PCA, then regression of different
% degrees without PCA

clear all;
close all;
load('../data/Data.mat');

proportion = 0.7;     
rows = size(Data,1); 
sep_idx = round(rows*proportion);
train = Data(1:sep_idx,:);
test = Data(sep_idx:end,:); %here we keep order because we wanna predict future values based on past values

%Targets
target_posx = PosX(1:sep_idx); % x position targets
target_posy = PosY(1:sep_idx); % y position 
target_posx_te = PosX(sep_idx:end); 
target_posy_te = PosY(sep_idx:end);


%Stdize
[std_train, mu, sigma] = zscore(train); 
std_test = (test - mu ) ./ sigma; %using same normalization coefficients 

%% PCA
[coeff, score, latent] = pca(std_train);
pca_train = std_train * coeff;
pca_test = std_test * coeff; %using same pca coefficients

%Choose PCs (+ graphs)
cum_Var=cumsum(latent)/sum(latent);
num_PC=1:length(latent);
figure;
plot(num_PC, cum_Var, 'r'); hold on;
xlabel('Number of PCs');
ylabel('Percentage of the total variance');
title('Total information carried by Principal Components');

idx90=find(cum_Var>0.9);
pc90=num_PC(idx90(1));
threshold90=line([pc90 pc90], [0 1]);
set(threshold90,'LineWidth',2,'color','blue');

chosen_PCs = 400;

FM_train = pca_train(:,1:chosen_PCs); % training feature matrix
I_train = ones(size(FM_train,1),1); % X should include a column of ones so that the model contains a constant term
X_train = [I_train FM_train];

%Regression and mse
bx = regress(target_posx, X_train); % coefficients b 
by = regress(target_posy, X_train);
x_hat = X_train * bx; %regression vectors
y_hat = X_train * by;
mse_posx = immse(target_posx, x_hat); %mse
mse_posy = immse(target_posy, y_hat);

FM_test = pca_test(:,1:chosen_PCs);
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

x_hat_te = X_test * bx; %using SAME coefficients
y_hat_te = X_test * by;
mse_posx_test = immse(target_posx_te, x_hat_te);
mse_posy_test = immse(target_posy_te, y_hat_te);

%% Train X and Y
figure
subplot(2,2,1)
plot(1:length(x_hat), target_posx, 'b'); hold on
plot(1:length(x_hat), x_hat, 'r'); hold off
ylabel('PosX')
xlabel('Time [ms]')
title('PosX coordinates - training')
legend('Target','Regressed')

subplot(2,2,2)
plot(1:length(y_hat), target_posy, 'b'); hold on
plot(1:length(y_hat), y_hat, 'r'); hold off 
ylabel('PosY')
xlabel('Time [ms]')
title('PosY coordinates - training')
legend('Target','Regressed')

%Test X and Y
subplot(2,2,3)
plot(1:length(x_hat_te), target_posx_te, 'b'); hold on
plot(1:length(x_hat_te), x_hat_te, 'g'); hold off 
ylabel('PosX')
xlabel('Time [ms]')
title('PosX coordinates - testing')
legend('Target','Regressed')

subplot(2,2,4)
plot(1:length(y_hat_te), target_posy_te, 'b'); hold on
plot(1:length(y_hat_te), y_hat_te, 'g'); hold off 
ylabel('PosY')
xlabel('Time [ms]')
title('PosY coordinates - testing')
legend('Target','Regressed')

%% Errors

mse_posx = immse(target_posx, x_hat); 
mse_posy = immse(target_posy, y_hat);
mse_posx_te = immse(target_posx_te, x_hat_te);
mse_posy_te = immse(target_posy_te, y_hat_te);

errors_linear = [mse_posx, mse_posy, mse_posx_te, mse_posy_te];
figure
bar(errors_linear);

%% Try for different polynomial regression
model_orders = [1:5];
%FM_train = std_train; % training feature matrix
I_train = ones(size(FM_train,1),1); 
X_train = [I_train];
%FM_test = std_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test];

order_errors_x = zeros(length(model_orders),1);
order_errors_x_te = zeros(length(model_orders),1);
order_errors_y = zeros(length(model_orders),1);
order_errors_y_te = zeros(length(model_orders),1);

for order_idx=1:1:length(model_orders)
    X_train = [X_train FM_train.^order_idx];
    X_test = [X_test FM_test.^order_idx];
    
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
    
    order_errors_x(order_idx) = mse_posx;
    order_errors_y(order_idx) = mse_posy;    

    order_errors_x_te(order_idx) = mse_posx_te;
    order_errors_y_te(order_idx) = mse_posy_te;
end

figure
subplot(1,2,1)
plot(model_orders, order_errors_x, 'LineWidth', 1.5); hold on
plot(model_orders, order_errors_x_te, 'LineWidth', 1.5); hold off
xlabel('Model order m');
ylabel('MSE');
title('PosX');
legend('Training','Testing');

subplot(1,2,2)
plot(model_orders, order_errors_y, 'LineWidth', 1.5); hold on
plot(model_orders, order_errors_y_te, 'LineWidth', 1.5); hold off
xlabel('Model order m');
ylabel('MSE');
title('PosY');
legend('Training','Testing');

%% Storage with and without PCA

errors = [order_errors_x order_errors_y order_errors_x_te order_errors_x_te];
%% No PCA

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
target_posx_te = PosX(sep_idx:end); 
target_posy_te = PosY(sep_idx:end);
FM_test = std_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

x_hat_te = X_test * bx; %using SAME coefficients
y_hat_te = X_test * by;
mse_posx_test = immse(target_posx_te, x_hat_te);
mse_posy_test = immse(target_posy_te, y_hat_te);

%Train X and Y
figure
subplot(2,2,1)
plot(1:length(x_hat), target_posx, '--b'); hold on
plot(1:length(x_hat), x_hat, 'r'); hold off
ylabel('PosX')
xlabel('Time [ms]')
title('PosX movements - training')
legend('Target','Regressed')

subplot(2,2,2)
plot(1:length(y_hat), target_posy, '--b'); hold on
plot(1:length(y_hat), y_hat, 'r'); hold off 
ylabel('PosY')
xlabel('Time [ms]')
title('PosY movements - training')
legend('Target','Regressed')

%Test X and Y
subplot(2,2,3)
plot(1:length(x_hat_te), target_posx_te, '--b'); hold on
plot(1:length(x_hat_te), x_hat_te, 'r'); hold off 
ylabel('PosX')
xlabel('Time [ms]')
title('PosX movements - testing')
legend('Target','Regressed')

subplot(2,2,4)
plot(1:length(y_hat_te), target_posy_te, '--b'); hold on
plot(1:length(y_hat_te), y_hat_te, 'r'); hold off 
ylabel('PosY')
xlabel('Time [ms]')
title('PosY movements - testing')
legend('Target','Regressed')

%Errors

mse_posx = immse(target_posx, x_hat); 
mse_posy = immse(target_posy, y_hat);
mse_posx_te = immse(target_posx_te, x_hat_te);
mse_posy_te = immse(target_posy_te, y_hat_te);
errors_linear_n = [mse_posx, mse_posy, mse_posx_te, mse_posy_te];
errors_linear_rmse = sqrt(errors_linear);
figure
bar(errors_linear_n);


%% With increasing order

model_orders = [1:5];
FM_train = std_train; % training feature matrix
I_train = ones(size(FM_train,1),1); 
X_train = [I_train];
FM_test = std_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test];

order_errors_x = zeros(length(model_orders),1);
order_errors_x_te = zeros(length(model_orders),1);
order_errors_y = zeros(length(model_orders),1);
order_errors_y_te = zeros(length(model_orders),1);

for order_idx=1:1:length(model_orders)
    X_train = [X_train FM_train.^order_idx];
    X_test = [X_test FM_test.^order_idx];
    
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
    
    order_errors_x(order_idx) = mse_posx;
    order_errors_y(order_idx) = mse_posy;    

    order_errors_x_te(order_idx) = mse_posx_te;
    order_errors_y_te(order_idx) = mse_posy_te;
end

figure
subplot(1,2,1)
plot(model_orders, order_errors_x, 'LineWidth', 1.5); hold on
plot(model_orders, order_errors_x_te, 'LineWidth', 1.5); hold off
xlabel('Model order m');
ylabel('MSE');
title('PosX');
legend('Training','Testing');

subplot(1,2,2)
plot(model_orders, order_errors_y, 'LineWidth', 1.5); hold on
plot(model_orders, order_errors_y_te, 'LineWidth', 1.5); hold off
xlabel('Model order m');
ylabel('MSE');
title('PosY');
legend('Training','Testing');


%% Plot PCA and no PCA
errors = [errors ; order_errors_x order_errors_y order_errors_x_te order_errors_y_te]

figure
%X test
subplot(1,2,1)
plot(model_orders, errors(1:5,3), 'LineWidth', 1.5); hold on 
plot(model_orders, errors(6:10,3), 'LineWidth', 1.5); hold off 
xlabel('Model order');
ylabel('MSE');
legend('With PCA', 'Without PCA');
title('PosX testing');

%Y test
subplot(1,2,2)
plot(model_orders, errors(1:5,4), 'LineWidth', 1.5); hold on 
plot(model_orders, errors(6:10,4), 'LineWidth', 1.5); hold off 
xlabel('Model order');
ylabel('MSE');
legend('With PCA', 'Without PCA');
title('PosY testing');



