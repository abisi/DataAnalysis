%% Guidesheet 7 : PCA and Regression -> we will have to com

clear all;
close all;
load('../data/Data.mat');

%% Data set partitioning and PCA 

%Partition
proportion = 0.7;     
rows = size(Data,1); %12862
sep_idx = round(rows*proportion);
train = Data(1:sep_idx,:);
test = Data(sep_idx:end,:); %here we keep order because we wanna predict future values based on past values



[std_train, mu, sigma] = zscore(train); %normalize on train data
std_test = (test - mu ) ./ sigma; %using same normalization coefficients for test data

%PCA

[coeff, score, latent] = pca(std_train);
%Coeff contains PC coeffs of original data.
%Each column of COEFF matrix contains coeffs for one PC.
%Columns are in descending order in terms of component variance
%pca centers data and uses singular value decomp algo by default

%Score is representaition of data in PC space, we reconstruct centered data
%using score*coeff'

%Latent contains PC variances, i.e. eigenvalues of cov. matrix of original
%data.

% Presumably this projects original data into PC space -> NOTE TO SELF : ASK Q ABOUT PCA
pca_train = std_train * coeff;
pca_test = std_test * coeff; %using same coefficients

%Choose PCs (+ graphs)
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

figure
bar(latent);

%% Regression - linear
chosen_PCs = 741;  %for 90% total variance
%Train
target_posx = PosX(1:sep_idx); % x position reg. targets
target_posy = PosY(1:sep_idx); % y position reg. targets
FM_train = pca_train(:,1:chosen_PCs); % training feature matrix
I_train = ones(size(FM_train,1),1); % X should include a column of ones so that the model contains a constant term
X_train = [I_train FM_train];

%Regression and mse
bx = regress(target_posx, X_train(:,1:chosen_PCs)); % coefficients b %JULIAN'S QUESTION : why do we select 1:chosen_PCs Here ; x_train only has 742 columns
by = regress(target_posy, X_train(:,1:chosen_PCs));
x_hat = X_train(:,1:chosen_PCs) * bx; %regression vectors
y_hat = X_train(:,1:chosen_PCs) * by;
mse_posx = immse(target_posx, x_hat); %mse
mse_posy = immse(target_posy, y_hat);

%Test
target_posx_test = PosX(sep_idx:end); 
target_posy_test = PosY(sep_idx:end);
FM_test = pca_test(:,1:chosen_PCs);
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

x_hat_te = X_test(:,1:chosen_PCs) * bx; %using SAME coefficients
y_hat_te = X_test(:,1:chosen_PCs) * by;
mse_posx_test = immse(target_posx_test, x_hat_te);
mse_posy_test = immse(target_posy_test, y_hat_te);

%Higher error on test set (expected)

%% Plot real and regressed vectors
%Motion X
regressed_x = [x_hat; x_hat_te];
figure
plot(regressed_x); hold on
plot(PosX); hold off
xlabel('Time (ms)')
ylabel('')
title('Predicted cartesian coordinate X of monkey''s wrist')

%Motion Y
regressed_y = [y_hat; y_hat_te];
figure
plot(regressed_y); hold on
plot(PosY); hold off
xlabel('Time (ms)')
ylabel('')
title('Predicted cartesian coordinate Y of monkey''s wrist')

%Rather good fits ! 

%% Regression - 2nd order polynomial regressor
%Second order polynomial data sets
X_train_2 = [I_train FM_train FM_train.^2];
X_test_2 = [I_test FM_test FM_test.^2];

%Train: regress, predict, mse
bx_2 = regress(target_posx, X_train_2); 
by_2 = regress(target_posy, X_train_2);
x_hat_2 = X_train_2 * bx_2; 
y_hat_2 = X_train_2 * by_2;
mse_posx_2 = immse(target_posx, x_hat_2); 
mse_posy_2 = immse(target_posy, y_hat_2);

%Test: regress, predict, mse
x_hat_te_2 = X_test_2 * bx_2; %using same coefficients
y_hat_te_2 = X_test_2 * by_2;
mse_posx_test_2 = immse(target_posx_test, x_hat_te_2);
mse_posy_test_2 = immse(target_posy_test, y_hat_te_2);

%Training errors: they decrease a bit
%Testing errors: they increase a bit

%% Plot real and regressed vectors - 2nd order polynomial regressor
%Motion X
regressed_x_2 = [x_hat_2; x_hat_te_2];
figure
plot(regressed_x_2); hold on
plot(PosX); hold off
xlabel('Time (ms)')
ylabel('')
title('Predicted cartesian coordinate X of monkey''s wrist')

%Motion Y
regressed_y_2 = [y_hat_2; y_hat_te_2];
figure
plot(regressed_y_2); hold on
plot(PosY); hold off
xlabel('Time (ms)')
ylabel('')
title('Predicted cartesian coordinate Y of monkey''s wrist')

%Also seems like a good fit with 2nd order. 
%However, there might an order M at which he will overfit the data !


%% Gradually include features

n_PCs = size(pca_train,2);

FM_train = pca_train;
I_train = ones(size(FM_train,1),1);
X_train = [I_train FM_train];

FM_test = pca_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

%Init. error vetors
%Train
error_x = zeros(n_PCs,1);
error_y = zeros(n_PCs,1);
error_x_2 = zeros(n_PCs,1);
error_y_2 = zeros(n_PCs,1);
%Test
error_x_te = zeros(n_PCs,1);
error_y_te = zeros(n_PCs,1);
error_x_2_te = zeros(n_PCs,1);
error_y_2_te = zeros(n_PCs,1);


for PC_idx=1:50:n_PCs
    disp(PC_idx)
    %First order coeff
    bx = regress(target_posx,X_train(:,1:PC_idx)); 
    by = regress(target_posy,X_train(:,1:PC_idx));
    %Second order coeff
    bx_2 = regress(target_posx,X_train_2(:,1:PC_idx)); 
    by_2 = regress(target_posy,X_train_2(:,1:PC_idx)); 
    
    %Predict
    x_hat = X_train(:,1:PC_idx) * bx; 
    y_hat = X_train(:,1:PC_idx) * by;
    x_hat_2 = X_train_2(:,1:PC_idx) * bx_2; 
    y_hat_2 = X_train_2(:,1:PC_idx) * by_2;
    
    x_hat_te = X_test(:,1:PC_idx) * bx; 
    y_hat_te = X_test(:,1:PC_idx) * by;
    x_hat_2_te = X_test_2(:,1:PC_idx) * bx_2; 
    y_hat_2_te = X_test_2(:,1:PC_idx) * by_2;
    
    %Errors
    error_x(PC_idx) = immse(target_posx, x_hat);
    error_y(PC_idx) = immse(target_posy, y_hat);
    error_x_2(PC_idx) = immse(target_posx, x_hat_2);
    error_y_2(PC_idx) = immse(target_posy, y_hat_2);

    error_x_te(PC_idx) = immse(target_posx_test, x_hat_te);
    error_y_te(PC_idx) = immse(target_posy_test, y_hat_te);
    error_x_2_te(PC_idx) = immse(target_posx_test, x_hat_2_te);
    error_y_2_te(PC_idx) = immse(target_posy_test, y_hat_2_te);
end



%% Let's plot the errors with increaseing number of PCs
x = [1:50:n_PCs]; %number PCs

figure;
subplot(2,2,1) 
plot(x, error_x(x), 'LineWidth', 1); hold on
plot(x, error_x_te(x), 'LineWidth', 1); 
xlabel('nb PCs');
ylabel('Error');
title('Linearly regressed PosX ');
legend('Train error','Test error');

subplot(2,2,2)
plot(x, error_y(x), 'LineWidth', 1); hold on
plot(x, error_y_te(x), 'LineWidth', 1);  
xlabel('nb PCs');
ylabel('Error');
title('Linearly regressed PosY ');
legend('Train error','Test error');


subplot(2,2,3)
plot(x, error_x_2(x), 'LineWidth', 1); hold on
plot(x, error_x_2_te(x), 'LineWidth', 1);  
xlabel('nb PCs');
ylabel('Error');
title('Polynomial regression of PosX ');
legend('Train error','Test error');


subplot(2,2,4)
plot(x, error_y_2(x), 'LineWidth', 1); hold on
plot(x, error_y_2_te(x), 'LineWidth', 1);
xlabel('nb PCs');
ylabel('Error');
title('Polynomial regression of PosY ');
legend('Train error','Test error');

